"""Measured SNLI development pilot, separate from reported cross-validation.

Run ``python -m part1.snli_pilot``. Notebook inspection uses the saved report
instead of retraining. Only eligible official train/validation rows enter the
pilot; the official test is not scored or used to select settings.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
from time import perf_counter
from typing import Any

from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from scipy import sparse
import torch

from .project_setup import DEFAULT_CONFIG, REPO_ROOT, prepare_workspace
from .snli_data import ID_COLUMNS, LABEL_NAMES
from .snli_evaluation import prepare_evaluation_protocol, select_complete_groups
from .snli_features import (
    PairFeatureEncoder, REPRESENTATIONS, _sha256, fit_pair_features,
    matrix_summary, tokenize_pairs,
)
from .snli_models import (
    CLASSIFIERS, FittedClassifier, fit_classifier, load_prediction_bundle,
    save_prediction_bundle, score_probabilities, select_input_features,
)


DEFAULT_PILOT = REPO_ROOT / "part1/configs/snli_pilot.json"
FROZEN_CONFIG = REPO_ROOT / "part1/configs/snli_project1_frozen.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _hash_tree(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): _sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def _verify(directory: Path, manifest: dict[str, Any]) -> None:
    for name, digest in manifest["artifact_sha256"].items():
        path = directory / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Pilot artifact changed or is missing: {path}")


class SampledMemory:
    """Sample whole-process RSS every 20 ms; not allocator-only/isolated model RAM."""

    def __enter__(self):
        self.process = psutil.Process()
        self.peak_bytes = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while not self.stop.wait(0.02):
            self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)

    def __exit__(self, *exception):
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def select_common_epoch(curves: pd.DataFrame, candidates: list[int], tolerance: float) -> tuple[int, pd.DataFrame]:
    """Choose the earliest near-best common schedule using independent pilot data."""
    selected = curves.loc[curves["epoch"].isin(candidates)]
    if selected.groupby("epoch").size().to_dict() != {epoch: 6 for epoch in candidates}:
        raise ValueError("Each epoch candidate needs all six neural-system observations.")
    if selected.groupby(["epoch", "representation", "classifier"]).size().ne(1).any():
        raise ValueError("Epoch candidates must contain six distinct neural systems.")
    table = selected.groupby("epoch", as_index=False).agg(
        mean_validation_accuracy=("validation_accuracy", "mean"),
        mean_validation_nll=("validation_nll", "mean"),
    )
    table["within_tolerance"] = table["mean_validation_accuracy"] >= table["mean_validation_accuracy"].max() - tolerance
    epoch = int(table.loc[table["within_tolerance"], "epoch"].min())
    table["selected"] = table["epoch"].eq(epoch)
    return epoch, table


def project_comparison_budget(
    fit_seconds: float, preprocessing_seconds: float, pilot_rows: int,
    full_rows: int, config: dict[str, Any], policy: dict[str, Any],
) -> tuple[int, pd.DataFrame]:
    """Rough linear projections with an explicit safety factor, not measured CV."""
    candidates = list(policy["comparison_row_candidates"])
    if policy["include_full_pool_candidate"]:
        candidates.append(full_rows)
    repetitions = len(config["seeds"]["cv_splits"])
    folds = config["cross_validation"]["num_folds"]
    rows = []
    for target in sorted(set(value for value in candidates if value <= full_rows)):
        scale = target * (folds - 1) / folds / pilot_rows
        estimate = (fit_seconds + preprocessing_seconds) * folds * repetitions * scale
        conservative = estimate * policy["runtime_projection_safety_factor"]
        rows.append({
            "target_rows": target, "estimated_primary_seconds": estimate,
            "conservative_primary_seconds": conservative,
            "within_budget": conservative <= policy["primary_matrix_budget_seconds"],
        })
    table = pd.DataFrame(rows)
    eligible = table.loc[table["within_budget"], "target_rows"]
    if not len(eligible):
        raise RuntimeError("No proposed comparison pool fits the pilot's planning budget; add a smaller candidate before freezing.")
    target = int(eligible.max())
    table["selected"] = table["target_rows"].eq(target)
    return target, table


def _prepare_data(workspace, protocol, policy, directory: Path):
    cache = directory / "features"
    marker = cache / "manifest.json"
    if marker.is_file():
        manifest = _read(marker)
        _verify(cache, manifest)
        encoder = PairFeatureEncoder.load(cache)
        frames = {name: pd.read_parquet(cache / f"{name}_rows.parquet") for name in ("train", "validation")}
        counts = {name: sparse.load_npz(cache / f"{name}_counts.npz").tocsr() for name in frames}
    else:
        cache.mkdir(exist_ok=True)
        started = perf_counter()
        candidates = {
            "train": protocol["cv_pool"],
            "validation": protocol["membership"].loc[protocol["membership"]["pilot_validation"]],
        }
        source = load_from_disk(protocol["manifest"]["prepared_dataset"])
        frames = {}
        for name, official in (("train", "train"), ("validation", "validation")):
            rows = select_complete_groups(candidates[name], policy[f"{'training' if name == 'train' else 'validation'}_target_rows"], policy["sampling_seed"])
            frame = source[official].select(rows["clean_index"].tolist()).to_pandas()
            if frame["example_id"].tolist() != rows["example_id"].tolist():
                raise ValueError("Pilot source rows are out of order.")
            frame["clean_index"] = rows["clean_index"].to_numpy()
            frame["premise_group"] = rows["premise_group"].to_numpy()
            frame.insert(0, "pilot_index", np.arange(len(frame)))
            frames[name] = frame
        if set(frames["train"]["premise"]) & set(frames["validation"]["premise"]):
            raise ValueError("Pilot training/validation premise leakage.")
        with SampledMemory() as memory:
            config = workspace["config"]
            encoder, tokens = fit_pair_features(
                frames["train"]["premise"].tolist(), frames["train"]["hypothesis"].tolist(),
                config["tokenizer"], config["bigram_features"],
            )
            validation_tokens = tokenize_pairs(
                encoder.tokenizer, frames["validation"]["premise"].tolist(), frames["validation"]["hypothesis"].tolist(),
            )
            counts = {"train": encoder.count_matrix(tokens), "validation": encoder.count_matrix(validation_tokens)}
            encoder.save(cache)
            tokens.save(cache / "train_tokens.npz")
            validation_tokens.save(cache / "validation_tokens.npz")
            for name in frames:
                frames[name].to_parquet(cache / f"{name}_rows.parquet", index=False)
                sparse.save_npz(cache / f"{name}_counts.npz", counts[name])
        manifest = {
            "preprocessing_seconds": perf_counter() - started,
            "sampled_peak_process_rss_bytes": memory.peak_bytes,
            "training_rows": len(frames["train"]), "validation_rows": len(frames["validation"]),
            "actual_vocab_size": encoder.vocab_size, "bigram_types": len(encoder.bigrams),
            "artifact_sha256": _hash_tree(cache),
        }
        _write(marker, manifest)
    matrices = {name: encoder.representations(matrix) for name, matrix in counts.items()}
    return encoder, frames, matrices, manifest


def _train_system(representation, kind, matrices, frames, encoder, config, policy, directory, verbose):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "training_result.json"
    if marker.is_file():
        result = _read(marker)
        _verify(directory, result)
        return result
    X, Xval = matrices["train"][representation], matrices["validation"][representation]
    y, yval = frames["train"]["label"].to_numpy(), frames["validation"]["label"].to_numpy()
    attempts = []
    with SampledMemory() as memory:
        if kind == "logistic":
            for maximum in policy["logistic_max_iter_candidates"]:
                trial = deepcopy(config)
                trial["logistic_training"]["max_iter"] = maximum
                classifier = fit_classifier(kind, X, y, trial)
                attempts.append({"max_iter": maximum, **classifier.fit_info})
                if classifier.fit_info["converged"]:
                    break
            if not classifier.fit_info["converged"]:
                _write(directory / "failed_convergence.json", {"attempts": attempts})
                raise RuntimeError(f"SAGA did not converge for {representation}; see {directory}.")
        else:
            candidates = directory / "epoch_candidates"
            candidates.mkdir(exist_ok=True)

            def observe(view: FittedClassifier):
                started = perf_counter()
                logp = view.predict_log_proba(Xval)
                scores = score_probabilities(yval, np.exp(logp))
                stable_nll = float(-logp[np.arange(len(yval)), yval].mean())
                delta = abs(stable_nll - scores["nll"])
                if delta > 1e-6:
                    raise ValueError("Pilot probability NLL and log-softmax objective disagree.")
                row = view.history[-1]
                row.update({
                    "validation_accuracy": scores["accuracy"], "validation_nll": scores["nll"],
                    "validation_macro_f1": scores["macro_f1"], "nll_objective_abs_difference": delta,
                    "validation_seconds": perf_counter() - started,
                })
                if row["epoch"] in policy["epoch_candidates"]:
                    torch.save(view.estimator.state_dict(), candidates / f"epoch_{row['epoch']}.pt")
                    if verbose:
                        print(f"  {representation}/{kind}, epoch {row['epoch']}: pilot val accuracy={scores['accuracy']:.4f}, NLL={scores['nll']:.4f}", flush=True)

            classifier = fit_classifier(kind, X, y, config, num_epochs=max(policy["epoch_candidates"]), epoch_observer=observe)
    save_prediction_bundle(directory / "max_schedule_bundle", classifier, encoder, representation, context={"stage": "pilot", "reported_cv_result": False})
    pd.DataFrame(classifier.history).to_csv(directory / "curves.csv", index=False)
    result = {
        "representation": representation, "classifier": kind,
        "sampled_peak_process_rss_bytes": memory.peak_bytes,
        "attempts": attempts, "fit_info": classifier.fit_info,
        "history": classifier.history, "parameters": classifier.parameter_count,
        "artifact_sha256": _hash_tree(directory),
    }
    _write(marker, result)
    return result


def _save_evaluation(classifier, encoder, representation, mode, frames, matrices, directory, context):
    Xval = select_input_features(matrices["validation"][representation], mode)
    yval = frames["validation"]["label"].to_numpy()
    started = perf_counter()
    probabilities = classifier.predict_proba(Xval)
    scores = score_probabilities(yval, probabilities)
    if not (directory / "bundle.json").is_file():
        save_prediction_bundle(directory, classifier, encoder, representation, input_mode=mode, context=context)
    reloaded = load_prediction_bundle(directory).predict_proba(
        frames["validation"]["premise"].tolist(), frames["validation"]["hypothesis"].tolist(),
    )
    delta = float(np.max(np.abs(probabilities - reloaded)))
    if delta > 1e-6:
        raise ValueError("Pilot bundle predictions changed after reload.")
    predictions = frames["validation"][["pilot_index", *ID_COLUMNS, "label"]].copy()
    predictions["prediction"] = probabilities.argmax(axis=1)
    for label, name in enumerate(LABEL_NAMES):
        predictions[f"p_{name}"] = probabilities[:, label]
    predictions["stage"] = "pilot_validation"
    predictions.to_parquet(directory / "predictions.parquet", index=False)
    _write(directory / "metrics.json", scores)
    return {
        "representation": representation, "classifier": classifier.kind, "input_mode": mode,
        "accuracy": scores["accuracy"], "nll": scores["nll"], "macro_f1": scores["macro_f1"],
        "training_rows": classifier.fit_info["training_rows"], "validation_rows": len(yval),
        "input_dim": classifier.input_dim, "parameters": classifier.parameter_count,
        "epochs": classifier.fit_info.get("epochs_completed"),
        "converged": classifier.fit_info.get("converged"),
        "training_seconds": classifier.fit_info["fit_seconds"],
        "evaluation_reload_seconds": perf_counter() - started,
        "reload_max_abs_difference": delta,
    }


def _plot_curves(curves, epoch, directory):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for column, representation in enumerate(REPRESENTATIONS):
        selected = curves.loc[curves["representation"].eq(representation)]
        for kind in CLASSIFIERS[1:]:
            rows = selected.loc[selected["classifier"].eq(kind)]
            axes[0, column].plot(rows["epoch"], rows["validation_accuracy"], marker="o", markersize=3, label=kind)
            axes[1, column].plot(rows["epoch"], rows["validation_nll"], marker="o", markersize=3, label=kind)
        axes[0, column].set_title(representation)
        for row in (0, 1):
            axes[row, column].axvline(epoch, color="grey", linestyle="--", label="Frozen common epoch")
            axes[row, column].grid(alpha=0.2)
        axes[1, column].set_xlabel("Pilot epoch")
    axes[0, 0].set_ylabel("Pilot validation accuracy")
    axes[1, 0].set_ylabel("Pilot validation NLL (nats)")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Development pilot curves — not reported cross-validation")
    fig.tight_layout()
    path = directory / "learning_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_pilot(
    config_path: str | Path = DEFAULT_CONFIG, policy_path: str | Path = DEFAULT_PILOT,
    *, frozen_config_path: str | Path = FROZEN_CONFIG, verbose: bool = False,
) -> dict[str, Any]:
    policy_path = REPO_ROOT / policy_path
    policy = _read(policy_path)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(policy["neural_threads"])
    try:
        return _run_pilot(config_path, policy, REPO_ROOT / frozen_config_path, verbose)
    finally:
        torch.set_num_threads(original_threads)


def _run_pilot(config_path, policy, frozen_path, verbose):
    started = perf_counter()
    workspace = prepare_workspace(config_path)
    config = deepcopy(workspace["config"])
    config["neural_training"]["num_threads"] = policy["neural_threads"]
    protocol = prepare_evaluation_protocol(config_path)
    specification = {
        "base_configuration": workspace["config"], "policy": policy,
        "source_protocol_id": protocol["manifest"]["protocol_id"],
        "implementation_sha256": _sha256(Path(__file__)),
        "model_code_sha256": _sha256(Path(__file__).with_name("snli_models.py")),
        "feature_code_sha256": _sha256(Path(__file__).with_name("snli_features.py")),
        "frozen_config_path": str(frozen_path.resolve()),
    }
    pilot_id = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:16]
    directory = workspace["paths"]["output"] / "pilot" / pilot_id
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "manifest.json"
    if marker.is_file():
        return _load_report(marker)
    _write(directory / "running.json", {"pilot_id": pilot_id, "specification": specification, "status": "incomplete"})
    if verbose:
        print(f"Pilot {pilot_id}: preparing complete-group samples and training-only features...", flush=True)
    encoder, frames, matrices, preprocessing = _prepare_data(workspace, protocol, policy, directory)
    if verbose:
        print(f"Actual pilot rows: {len(frames['train']):,} train / {len(frames['validation']):,} validation; V={encoder.vocab_size}, B={len(encoder.bigrams)}.", flush=True)
    systems = []
    for representation in REPRESENTATIONS:
        for kind in CLASSIFIERS:
            if verbose:
                print(f"Training pilot {representation}/{kind}...", flush=True)
            systems.append(_train_system(
                representation, kind, matrices, frames, encoder, config, policy,
                directory / "models" / f"{representation}__{kind}", verbose,
            ))
    curves = pd.concat([
        pd.DataFrame(result["history"]).assign(representation=result["representation"], classifier=result["classifier"])
        for result in systems if result["classifier"] != "logistic"
    ], ignore_index=True)
    selected_epoch, selection = select_common_epoch(curves, policy["epoch_candidates"], config["evaluation"]["rope_half_width"])
    curves.to_csv(directory / "curves.csv", index=False)
    selection.to_csv(directory / "epoch_selection.csv", index=False)
    final_results, projected_fit_seconds = [], 0.0
    for result in systems:
        representation, kind = result["representation"], result["classifier"]
        source = directory / "models" / f"{representation}__{kind}"
        classifier = load_prediction_bundle(source / "max_schedule_bundle").classifier
        if kind != "logistic":
            classifier.estimator.load_state_dict(torch.load(source / "epoch_candidates" / f"epoch_{selected_epoch}.pt", map_location="cpu", weights_only=True))
            classifier.settings["num_epochs"] = selected_epoch
            classifier.history = classifier.history[:selected_epoch]
            classifier.fit_info.update({
                "epochs_completed": selected_epoch, "checkpoint_policy": "pilot_validation_selected_common_epoch",
                "full_pilot_fit_seconds": classifier.fit_info["fit_seconds"],
                "fit_seconds": sum(row["seconds"] for row in classifier.history),
            })
        projected_fit_seconds += classifier.fit_info["fit_seconds"]
        destination = directory / "selected_models" / f"{representation}__{kind}"
        row = _save_evaluation(classifier, encoder, representation, "pair", frames, matrices, destination, {"stage": "pilot", "pilot_id": pilot_id, "reported_cv_result": False})
        row["sampled_peak_process_rss_bytes"] = result["sampled_peak_process_rss_bytes"]
        final_results.append(row)
    target_rows, projections = project_comparison_budget(
        projected_fit_seconds, preprocessing["preprocessing_seconds"], len(frames["train"]),
        len(protocol["cv_pool"]), config, policy,
    )
    projections.to_csv(directory / "runtime_projection.csv", index=False)
    # Baselines use exactly the same pilot row order and preprocessing.
    diagnostic = config["diagnostics"]["hypothesis_only"]
    logistic_maximum = max(result["attempts"][-1]["max_iter"] for result in systems if result["classifier"] == "logistic")
    for representation, kind, mode in (
        ("unigram_counts", "class_prior", "pair"),
        (diagnostic["representation"], diagnostic["classifier"], diagnostic["input_mode"]),
    ):
        trial = deepcopy(config)
        with SampledMemory() as memory:
            for maximum in [value for value in policy["logistic_max_iter_candidates"] if value >= logistic_maximum]:
                trial["logistic_training"]["max_iter"] = maximum
                classifier = fit_classifier(kind, select_input_features(matrices["train"][representation], mode), frames["train"]["label"].to_numpy(), trial)
                if kind != "logistic" or classifier.fit_info["converged"]:
                    break
            if kind == "logistic" and not classifier.fit_info["converged"]:
                raise RuntimeError("Hypothesis-only SAGA did not converge within the pilot grid.")
            if kind == "logistic":
                logistic_maximum = max(logistic_maximum, maximum)
            row = _save_evaluation(
                classifier, encoder, representation, mode, frames, matrices,
                directory / "selected_models" / f"{representation}__{kind}__{mode}",
                {"stage": "pilot", "pilot_id": pilot_id, "reported_cv_result": False},
            )
        row["sampled_peak_process_rss_bytes"] = memory.peak_bytes
        final_results.append(row)
    results = pd.DataFrame(final_results)
    results.to_csv(directory / "metrics.csv", index=False)
    matrix_summary(matrices).to_csv(directory / "matrix_summary.csv", index=False)
    class_counts = pd.concat([
        frame.groupby("label").size().rename("rows").reset_index().assign(partition=name)
        for name, frame in frames.items()
    ], ignore_index=True)
    class_counts.to_csv(directory / "class_counts.csv", index=False)
    _plot_curves(curves, selected_epoch, directory)

    frozen = deepcopy(config)
    frozen["status"] = "frozen_after_pilot_before_reported_cv"
    frozen["neural_training"]["num_epochs"] = selected_epoch
    frozen["logistic_training"]["max_iter"] = logistic_maximum
    if target_rows < len(protocol["cv_pool"]):
        frozen["cross_validation"].update({
            "pool": "fixed_group_subset", "subset": {"target_rows": target_rows, "seed": policy["sampling_seed"]},
        })
    frozen["pilot_selection"] = {"pilot_id": pilot_id, "pilot_manifest": str(marker), "epoch_selection_rule": policy["epoch_selection_rule"]}
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    if frozen_path.exists() and _read(frozen_path) != frozen:
        raise FileExistsError(f"A different frozen configuration already exists: {frozen_path}")
    _write(frozen_path, frozen)
    if verbose:
        print(f"Freezing {selected_epoch} epochs and target {target_rows:,} CV rows; preparing its separate saved folds...", flush=True)
    final_protocol = prepare_evaluation_protocol(frozen_path, verbose=verbose)
    final_protocol["pool_summary"].to_csv(directory / "comparison_pool_summary.csv", index=False)
    final_protocol["fold_summary"].to_csv(directory / "comparison_fold_summary.csv", index=False)
    (directory / "running.json").unlink()
    manifest = {
        "stage": "development_pilot", "status": "complete", "pilot_id": pilot_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "specification": specification,
        "training_rows": len(frames["train"]), "validation_rows": len(frames["validation"]),
        "training_groups": frames["train"]["premise_group"].nunique(), "validation_groups": frames["validation"]["premise_group"].nunique(),
        "preprocessing": preprocessing, "selected_epochs": selected_epoch, "comparison_target_rows": target_rows,
        "comparison_actual_rows": len(final_protocol["cv_pool"]),
        "comparison_protocol_manifest": str(final_protocol["manifest_path"]),
        "frozen_configuration": {"path": str(frozen_path.resolve()), "sha256": _sha256(frozen_path)},
        "repository": workspace["manifest"]["repository"], "packages": workspace["manifest"]["packages"],
        "elapsed_seconds": perf_counter() - started,
        "checks": {
            "complete_premise_groups": True, "disjoint_training_validation_premises": True,
            "training_only_preprocessing": True, "finite_normalized_probabilities": True,
            "nll_agrees_with_log_softmax_objective": True, "bundles_reproduce_predictions": True,
            "new_comparison_folds_checked": True, "test_not_scored": True,
        },
        "limitations": [
            "Pilot validation is used for tuning; these scores are not reported CV or test estimates.",
            "Runtime projections assume linear scaling with a safety factor; actual solver/optimization/IO costs can differ.",
            "Projected primary cost excludes diagnostics and final refitting; RSS is sampled whole-process memory, not isolated model allocations.",
        ],
        "artifact_sha256": _hash_tree(directory),
    }
    _write(marker, manifest)
    _write(workspace["paths"]["manifests"] / "pilot.json", {"pilot_manifest": str(marker), "pilot_id": pilot_id, "sha256": _sha256(marker)})
    return _load_report(marker)


def _load_report(marker: Path) -> dict[str, Any]:
    manifest = _read(marker)
    _verify(marker.parent, manifest)
    frozen = manifest["frozen_configuration"]
    if _sha256(Path(frozen["path"])) != frozen["sha256"]:
        raise ValueError("The frozen pilot-selected configuration changed.")
    directory = marker.parent
    names = ("metrics", "curves", "epoch_selection", "runtime_projection", "class_counts", "matrix_summary", "comparison_pool_summary", "comparison_fold_summary")
    return {
        "manifest": manifest, "manifest_path": marker, "pilot_dir": directory,
        "frozen_config": _read(Path(frozen["path"])),
        "tables": {name: pd.read_csv(directory / f"{name}.csv") for name in names},
        "learning_curve_figure": directory / "learning_curves.png",
    }


def load_pilot_report(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Inspect saved artifacts without repeating training or changing selection."""
    config = _read(REPO_ROOT / config_path)
    pointer = REPO_ROOT / config["paths"]["output_dir"] / "manifests/pilot.json"
    if not pointer.is_file():
        raise FileNotFoundError("Run python -m part1.snli_pilot before inspecting notebook section 6.")
    record = _read(pointer)
    marker = Path(record["pilot_manifest"])
    if _sha256(marker) != record["sha256"]:
        raise ValueError("Pilot manifest changed.")
    return _load_report(marker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--frozen-config", type=Path, default=FROZEN_CONFIG)
    arguments = parser.parse_args()
    report = run_pilot(arguments.config, arguments.policy, frozen_config_path=arguments.frozen_config, verbose=True)
    print(report["tables"]["metrics"].to_string(index=False))
    print("\nSelected epochs:", report["manifest"]["selected_epochs"])
    print("Actual comparison pool:", report["manifest"]["comparison_actual_rows"])
    print("Frozen configuration:", report["manifest"]["frozen_configuration"]["path"])
    print("Pilot manifest:", report["manifest_path"])
