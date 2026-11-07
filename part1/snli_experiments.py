"""Resumable paired SNLI cross-validation using the pilot-frozen recipe.

Run ``python -m part1.snli_experiments``. Notebook inspection calls
load_cv_report(), which never trains models. Outer held-out rows are scored
only after fitting; they never select epochs, hyperparameters or checkpoints.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter, time_ns
import traceback
from typing import Any

from datasets import load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .project_setup import REPO_ROOT, prepare_workspace
from .snli_data import ID_COLUMNS, LABEL_NAMES
from .snli_evaluation import ROW_COLUMNS, prepare_evaluation_protocol
from .snli_features import REPRESENTATIONS, _sha256, prepare_fold_features
from .snli_models import (
    CLASSIFIERS, fit_classifier, load_prediction_bundle, save_prediction_bundle,
    score_probabilities, select_input_features, validate_probabilities,
)
from .snli_pilot import FROZEN_CONFIG, SampledMemory


CV_VERSION = "v1"
PROBABILITY_COLUMNS = [f"p_{name}" for name in LABEL_NAMES]
PREDICTION_ROWS = [*ROW_COLUMNS, "partition"]


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value) -> None:
    """Commit a small state file atomically, including after an interruption."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_tree(directory: Path) -> dict[str, str]:
    return {str(p.relative_to(directory)): _sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def _verify(directory: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"CV artifact changed or is missing: {path}")


@contextmanager
def _lock(directory: Path):
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is already running this CV experiment.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def planned_systems(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Nine primary systems and two fixed baselines, with unique stable IDs."""
    if config["representations"] != list(REPRESENTATIONS) or config["classifiers"] != list(CLASSIFIERS):
        raise ValueError("CV requires the three current representations and three model families.")
    if config["status"] != "frozen_after_pilot_before_reported_cv":
        raise ValueError("Use the pilot-frozen configuration for reported CV.")
    epochs = config["neural_training"]["num_epochs"]
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Reported CV needs a positive frozen epoch count.")
    systems = [
        {"representation": representation, "classifier": kind, "input_mode": "pair", "primary": True}
        for representation in REPRESENTATIONS for kind in CLASSIFIERS
    ]
    systems.append({"representation": "unigram_counts", "classifier": "class_prior", "input_mode": "pair", "primary": False})
    diagnostic = config["diagnostics"]["hypothesis_only"]
    systems.append({key: diagnostic[key] for key in ("representation", "classifier", "input_mode")} | {"primary": False})
    for system in systems:
        system["system_id"] = "__".join(system[key] for key in ("representation", "classifier", "input_mode"))
    if len({s["system_id"] for s in systems}) != 11:
        raise ValueError("Expected eleven distinct primary/baseline systems.")
    return systems


def validate_predictions(predictions: pd.DataFrame, rows: pd.DataFrame) -> dict[str, Any]:
    """Check exact held-out IDs/order/labels and score saved probability columns."""
    if not set([*PREDICTION_ROWS, "prediction", "stage", *PROBABILITY_COLUMNS]).issubset(predictions.columns):
        raise ValueError("Saved CV predictions have missing provenance/probability columns.")
    if not predictions[PREDICTION_ROWS].reset_index(drop=True).equals(rows[PREDICTION_ROWS].reset_index(drop=True)):
        raise ValueError("Saved CV predictions have wrong row order, IDs, groups or gold labels.")
    if predictions["example_id"].duplicated().any() or not predictions["stage"].eq("outer_cv_heldout").all():
        raise ValueError("CV predictions must contain each held-out example once with the correct stage.")
    if not predictions["source_split"].eq("train").all() or not predictions["partition"].eq("heldout").all():
        raise ValueError("Reported CV must use only held-out rows from the official training split.")
    probabilities = validate_probabilities(predictions[PROBABILITY_COLUMNS].to_numpy(), len(rows))
    if not np.array_equal(predictions["prediction"], probabilities.argmax(axis=1)):
        raise ValueError("Saved decisions disagree with the frozen argmax/tie convention.")
    return score_probabilities(rows["label"].to_numpy(), probabilities)


def _load_unit(directory: Path, identity: dict[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    result = _read(directory / "complete.json")
    if result["status"] != "complete" or result["identity"] != identity:
        raise ValueError("Completed CV unit has a different fold/configuration/system identity.")
    attempt = directory / result["attempt"]
    _verify(attempt, result["artifact_sha256"])
    scores = validate_predictions(pd.read_parquet(attempt / "predictions.parquet"), rows)
    row = result["metrics"]
    for name in ("accuracy", "nll", "macro_f1"):
        if not np.isclose(row[name], scores[name], rtol=0, atol=1e-12):
            raise ValueError("Saved CV metrics disagree with the checked held-out predictions.")
    return result


def _fit_unit(directory, identity, system, features, heldout_text, config, feature_seconds):
    directory.mkdir(parents=True, exist_ok=True)
    # Attempts are append-only. A killed or failed attempt is never overwritten.
    attempt = directory / f"attempt_{time_ns()}"
    attempt.mkdir()
    state = {"status": "running", "started_at_utc": _now(), "identity": identity}
    _write(attempt / "state.json", state)
    started = perf_counter()
    try:
        representation, mode = system["representation"], system["input_mode"]
        train, heldout = features["rows"]["train"], features["rows"]["heldout"]
        X = select_input_features(features["matrices"]["train"][representation], mode)
        Xheldout = select_input_features(features["matrices"]["heldout"][representation], mode)
        with SampledMemory() as memory:
            # No held-out labels or observer enter training. Retain the last fixed epoch.
            classifier = fit_classifier(system["classifier"], X, train["label"].to_numpy(), config)
            evaluation_started = perf_counter()
            probabilities = classifier.predict_proba(Xheldout)
            scores = score_probabilities(heldout["label"].to_numpy(), probabilities)
            predictions = heldout[PREDICTION_ROWS].copy()
            predictions["prediction"] = probabilities.argmax(axis=1)
            for label, name in enumerate(PROBABILITY_COLUMNS):
                predictions[name] = probabilities[:, label]
            predictions["stage"] = "outer_cv_heldout"
            predictions["run_id"] = identity["run_id"]
            predictions["system_id"] = identity["system_id"]
            predictions["seed"] = identity["seed"]
            predictions["fold"] = identity["fold"]
            validate_predictions(predictions, heldout)
            predictions.to_parquet(attempt / "predictions.parquet", index=False)
            _write(attempt / "scores.json", scores)
            save_prediction_bundle(attempt / "bundle", classifier, features["encoder"], representation,
                                   input_mode=mode, context={**identity, "stage": "reported_cv", "test_scored": False})
            # Full raw-text held-out reload, not just a sample of rows.
            loaded = load_prediction_bundle(attempt / "bundle")
            reproduced = loaded.predict_proba(heldout_text["premise"].tolist(), heldout_text["hypothesis"].tolist())
            difference = float(np.max(np.abs(probabilities - reproduced)))
            if difference > 1e-6:
                raise ValueError("CV probabilities changed after raw-text bundle reload.")
            evaluation_seconds = perf_counter() - evaluation_started
            pd.DataFrame(classifier.history).to_csv(attempt / "history.csv", index=False)
            _write(attempt / "fit_info.json", classifier.fit_info)
            model_bytes = sum(p.stat().st_size for p in (attempt / "bundle").iterdir() if p.suffix in (".pt", ".joblib", ".npy"))
            row = {
                **identity, **system, "training_rows": len(train), "heldout_rows": len(heldout),
                "model_seed": config["seeds"]["model"], "input_dim": X.shape[1],
                "training_nonzeros": X.nnz, "heldout_nonzeros": Xheldout.nnz,
                "training_csr_bytes": X.data.nbytes + X.indices.nbytes + X.indptr.nbytes,
                "parameters": classifier.parameter_count, "model_bytes": model_bytes,
                "accuracy": scores["accuracy"], "nll": scores["nll"], "macro_f1": scores["macro_f1"],
                "training_seconds": classifier.fit_info["fit_seconds"],
                "evaluation_reload_seconds": evaluation_seconds, "feature_preparation_seconds": feature_seconds,
                "epochs": classifier.fit_info.get("epochs_completed"),
                "converged": classifier.fit_info.get("converged"),
                "iterations": max(classifier.fit_info.get("iterations", [0])),
                "convergence_warnings": classifier.fit_info.get("convergence_warnings", []),
                "reload_max_abs_difference": difference,
                "checkpoint": str(attempt / "bundle"), "predictions": str(attempt / "predictions.parquet"),
                "feature_id": features["manifest"]["feature_id"],
                "feature_manifest": str(features["manifest_path"]),
                "feature_manifest_sha256": _sha256(features["manifest_path"]),
            }
        row["sampled_peak_process_rss_bytes"] = memory.peak_bytes
        row["unit_wall_seconds"] = perf_counter() - started
        _write(attempt / "row.json", row)
        _write(attempt / "state.json", {**state, "status": "complete", "finished_at_utc": _now()})
        result = {"status": "complete", "identity": identity, "attempt": attempt.name,
                  "metrics": row, "artifact_sha256": _hash_tree(attempt)}
        _write(directory / "complete.json", result)
        return result
    except BaseException as error:
        _write(attempt / "state.json", {**state, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
               "finished_at_utc": _now(), "error_type": type(error).__name__, "error": str(error),
               "traceback": traceback.format_exc()})
        raise


def _prepare_features(config_path, protocol, seed, fold, verbose):
    try:
        return prepare_fold_features(config_path, seed=seed, fold=fold, verbose=verbose)
    except ValueError as error:
        # Recover only the helper's explicit incomplete-write case. Corruption
        # of a completed/checksummed cache remains an error and is not hidden.
        match = re.fullmatch(r"Incomplete feature cache: (.+)\. Existing files were not overwritten\.", str(error))
        if not match:
            raise
        path = Path(match.group(1)).resolve()
        config = _read(REPO_ROOT / config_path)
        expected = REPO_ROOT / config["paths"]["output_dir"] / "features" / protocol["manifest"]["protocol_id"] / f"seed_{seed}_fold_{fold}"
        if path.parent != expected.resolve() or not re.fullmatch("[0-9a-f]{16}", path.name):
            raise
        destination = path.parent / "interrupted" / f"{path.name}_{time_ns()}"
        destination.parent.mkdir(exist_ok=True)
        path.rename(destination)
        if verbose:
            print(f"Preserved incomplete feature cache at {destination}; rebuilding this fold.", flush=True)
        return prepare_fold_features(config_path, seed=seed, fold=fold, verbose=verbose)


def summarize_metrics(metrics: pd.DataFrame, systems, seeds, folds: int) -> pd.DataFrame:
    expected = {(system["system_id"], seed, fold) for system in systems for seed in seeds for fold in range(folds)}
    keys = list(zip(metrics["system_id"], metrics["seed"], metrics["fold"]))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("A complete paired comparison needs every system on exactly the same saved folds.")
    summary = metrics.groupby(["system_id", "representation", "classifier", "input_mode", "primary"], as_index=False).agg(
        folds=("fold", "size"), accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"), accuracy_max=("accuracy", "max"),
        nll_mean=("nll", "mean"), nll_std=("nll", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        training_seconds_total=("training_seconds", "sum"), training_seconds_mean=("training_seconds", "mean"),
        parameters_min=("parameters", "min"), parameters_max=("parameters", "max"),
        input_dim_min=("input_dim", "min"), input_dim_max=("input_dim", "max"),
        model_bytes_mean=("model_bytes", "mean"),
    )
    return summary.sort_values(["primary", "accuracy_mean", "system_id"], ascending=[False, False, True]).reset_index(drop=True)


def _plots(metrics, summary, curves, directory):
    names = summary["system_id"].tolist()
    labels = [name.replace("__pair", "").replace("__", " / ") for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(metrics))
    positions = metrics["system_id"].map({name: i for i, name in enumerate(names)}).to_numpy()
    for ax, score, title in zip(axes, ("accuracy", "nll"), ("Held-out accuracy", "Held-out NLL (nats)")):
        ax.scatter(metrics[score], positions + jitter, s=16, alpha=.55, label="Saved fold scores")
        ax.scatter(summary[f"{score}_mean"], np.arange(len(names)), marker="D", color="black", s=28, label="Mean of folds")
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=.25)
    axes[0].set_yticks(np.arange(len(names)), labels)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=8)
    fig.suptitle("Paired repeated grouped CV — overlapping folds are not independent")
    fig.tight_layout()
    fig.savefig(directory / "fold_scores.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for column, representation in enumerate(REPRESENTATIONS):
        for kind in CLASSIFIERS[1:]:
            rows = curves.loc[(curves["representation"] == representation) & (curves["classifier"] == kind)]
            grouped = rows.groupby("epoch")
            for line, metric in enumerate(("train_nll_online", "train_accuracy_online")):
                means, stds = grouped[metric].mean(), grouped[metric].std().fillna(0)
                axes[line, column].plot(means.index, means, marker="o", label=kind)
                axes[line, column].fill_between(means.index, means - stds, means + stds, alpha=.15)
                axes[line, column].grid(alpha=.2)
        axes[0, column].set_title(representation)
        axes[1, column].set_xlabel("Fixed training epoch")
    axes[0, 0].set_ylabel("Online training NLL")
    axes[1, 0].set_ylabel("Online training accuracy")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Training histories: fold mean ± descriptive SD; no outer-fold checkpoint selection")
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)


def run_cross_validation(config_path: str | Path = FROZEN_CONFIG, *, max_fits: int | None = None, verbose: bool = False):
    """Run/resume all fits; max_fits bounds new work while keeping status incomplete."""
    if max_fits is not None and (not isinstance(max_fits, int) or isinstance(max_fits, bool) or max_fits < 1):
        raise ValueError("max_fits must be a positive integer or None.")
    config_path = REPO_ROOT / config_path
    workspace = prepare_workspace(config_path)
    config = workspace["config"]
    systems = planned_systems(config)
    protocol = prepare_evaluation_protocol(config_path, verbose=verbose)
    code_names = ("snli_experiments.py", "snli_models.py", "snli_features.py", "smlp.py", "experiment.py", "snli_evaluation.py", "project_setup.py")
    specification = {
        "cv_version": CV_VERSION, "configuration": config, "configuration_sha256": _sha256(config_path),
        "protocol_id": protocol["manifest"]["protocol_id"], "protocol_sha256": _sha256(protocol["manifest_path"]),
        "source_sha256": {name: _sha256(Path(__file__).with_name(name)) for name in code_names},
        "packages": workspace["manifest"]["packages"], "systems": systems,
        "checkpoint_policy": "last_fixed_epoch_no_outer_holdout_selection",
    }
    run_id = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:16]
    directory = workspace["paths"]["output"] / "cross_validation" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    original_threads = torch.get_num_threads()
    torch.set_num_threads(config["neural_training"]["num_threads"])
    try:
        with _lock(directory):
            return _run(directory, run_id, specification, workspace, protocol, systems, config_path, max_fits, verbose)
    finally:
        torch.set_num_threads(original_threads)


def _run(directory, run_id, specification, workspace, protocol, systems, config_path, max_fits, verbose):
    started = perf_counter()
    config = workspace["config"]
    marker = directory / "manifest.json"
    pointer = workspace["paths"]["manifests"] / "cross_validation.json"
    if marker.is_file():
        previous = _read(marker)
        if previous["specification"] != specification:
            raise ValueError("Saved CV specification differs from this run.")
        if previous["status"] == "complete":
            _write(pointer, {"run_id": run_id, "manifest_path": str(marker)})
            return _load_report(marker)
    else:
        previous = {}
    expected = len(systems) * len(config["seeds"]["cv_splits"]) * config["cross_validation"]["num_folds"]
    manifest = {
        "stage": "reported_cross_validation", "status": "running", "run_id": run_id,
        "created_at_utc": previous.get("created_at_utc", _now()), "specification": specification,
        "configuration_path": str(config_path), "protocol_manifest": str(protocol["manifest_path"]),
        "expected_units": expected, "completed_units": previous.get("completed_units", 0),
        "repository": workspace["manifest"]["repository"], "hardware": workspace["manifest"]["hardware"],
        "python": workspace["manifest"]["python"], "execution_history": previous.get("execution_history", []),
    }
    invocation = {"started_at_utc": _now(), "new_fits": 0, "reused_fits": 0}
    manifest["execution_history"].append(invocation)
    _write(marker, manifest)
    _write(pointer, {"run_id": run_id, "manifest_path": str(marker)})
    rows, curves, unit_hashes = [], [], {}
    try:
        source = None
        for repetition, seed in enumerate(config["seeds"]["cv_splits"]):
            for fold in range(config["cross_validation"]["num_folds"]):
                heldout = protocol["assignments"].loc[protocol["assignments"][f"fold_seed_{seed}"].eq(fold), ROW_COLUMNS].reset_index(drop=True)
                heldout["partition"] = "heldout"
                features, text = None, None
                for system in systems:
                    identity = {"run_id": run_id, "protocol_id": specification["protocol_id"],
                                "config_sha256": specification["configuration_sha256"], "dataset_revision": config["dataset"]["revision"],
                                "system_id": system["system_id"], "repetition": repetition, "seed": seed, "fold": fold}
                    unit = directory / "units" / f"seed_{seed}" / f"fold_{fold}" / system["system_id"]
                    complete = unit / "complete.json"
                    if complete.is_file():
                        result = _load_unit(unit, identity, heldout)
                        invocation["reused_fits"] += 1
                    else:
                        if max_fits is not None and invocation["new_fits"] >= max_fits:
                            manifest["status"] = "incomplete"
                            invocation.update({"finished_at_utc": _now(), "wall_seconds": perf_counter() - started, "status": "bounded_stop"})
                            _write(marker, manifest)
                            return {"manifest": manifest, "manifest_path": marker, "run_dir": directory}
                        if features is None:
                            feature_started = perf_counter()
                            features = _prepare_features(config_path, protocol, seed, fold, verbose)
                            feature_seconds = perf_counter() - feature_started
                            if not features["rows"]["heldout"][PREDICTION_ROWS].equals(heldout[PREDICTION_ROWS]):
                                raise ValueError("Feature row order disagrees with the frozen protocol.")
                            if source is None:
                                source = load_from_disk(protocol["manifest"]["prepared_dataset"])["train"]
                            text = source.select(heldout["clean_index"].tolist()).select_columns(["premise", "hypothesis"]).to_pandas()
                        if verbose:
                            print(f"CV {run_id}: fit {len(rows)+1}/{expected}, seed {seed}, fold {fold}, {system['system_id']}", flush=True)
                        result = _fit_unit(unit, identity, system, features, text, config, feature_seconds)
                        invocation["new_fits"] += 1
                        if verbose:
                            r = result["metrics"]
                            print(f"  accuracy={r['accuracy']:.4f}, NLL={r['nll']:.4f}, train={r['training_seconds']:.2f}s, converged={r['converged']}", flush=True)
                    rows.append(result["metrics"])
                    history_path = unit / result["attempt"] / "history.csv"
                    if system["classifier"] in CLASSIFIERS[1:]:
                        curves.append(pd.read_csv(history_path).assign(**identity, representation=system["representation"], classifier=system["classifier"]))
                    unit_hashes[str(complete.relative_to(directory))] = _sha256(complete)
                    manifest["completed_units"] = len(rows)
                    _write(marker, manifest)
                del features, text
                gc.collect()
        metrics = pd.DataFrame(rows).sort_values(["repetition", "fold", "system_id"]).reset_index(drop=True)
        summary = summarize_metrics(metrics, systems, config["seeds"]["cv_splits"], config["cross_validation"]["num_folds"])
        histories = pd.concat(curves, ignore_index=True)
        metrics.to_csv(directory / "fold_metrics.csv", index=False)
        summary.to_csv(directory / "summary.csv", index=False)
        histories.to_csv(directory / "learning_curves.csv", index=False)
        metrics.pivot(index=["repetition", "seed", "fold"], columns="system_id", values="accuracy").sort_index().to_csv(directory / "paired_accuracy.csv")
        convergence = metrics.loc[metrics["classifier"].eq("logistic"), ["system_id", "repetition", "seed", "fold", "converged", "iterations", "convergence_warnings"]]
        convergence.to_csv(directory / "convergence.csv", index=False)
        _plots(metrics, summary, histories, directory)
        manifest.update({
            "status": "complete", "finished_at_utc": _now(), "completed_units": len(metrics),
            "primary_fits": int(metrics["primary"].sum()), "baseline_fits": int((~metrics["primary"]).sum()),
            "nonconverged_logistic_fits": int(convergence["converged"].eq(False).sum()),
            "checks": {"matching_complete_fold_grid": True, "original_ids_and_gold_order_checked": True,
                       "finite_normalized_probabilities": True, "raw_text_bundle_reloads_match": True,
                       "fixed_training_length_no_outer_selection": True, "test_not_scored": True},
            "limitations": ["Fold means and SDs are descriptive; overlapping repeated folds are dependent.",
                            "Online training curves use changing weights/dropout, not end-of-epoch evaluation.",
                            "CV uses the fixed premise-group subset, not full-SNLI training; test remains unscored."],
        })
        invocation.update({"finished_at_utc": _now(), "wall_seconds": perf_counter() - started, "status": "complete"})
        manifest["artifact_sha256"] = unit_hashes | {
            p.name: _sha256(p) for p in directory.iterdir() if p.suffix in (".csv", ".png")
        }
        _write(marker, manifest)
        return _load_report(marker)
    except BaseException as error:
        invocation.update({"finished_at_utc": _now(), "wall_seconds": perf_counter() - started,
                           "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           "error_type": type(error).__name__, "error": str(error)})
        manifest["status"] = invocation["status"]
        _write(marker, manifest)
        raise


def _load_report(marker: Path) -> dict[str, Any]:
    manifest = _read(marker)
    if manifest["status"] != "complete":
        raise ValueError(f"CV is {manifest['status']}: {manifest['completed_units']}/{manifest['expected_units']} units complete. Resume the runner.")
    _verify(marker.parent, manifest["artifact_sha256"])
    for name in manifest["artifact_sha256"]:
        if name.endswith("/complete.json"):
            unit = marker.parent / name
            result = _read(unit)
            _verify(unit.parent / result["attempt"], result["artifact_sha256"])
    config_path = Path(manifest["configuration_path"])
    if _sha256(config_path) != manifest["specification"]["configuration_sha256"]:
        raise ValueError("The frozen CV configuration changed.")
    directory = marker.parent
    return {"manifest": manifest, "manifest_path": marker, "run_dir": directory,
            "tables": {name: pd.read_csv(directory / f"{name}.csv") for name in ("fold_metrics", "summary", "learning_curves", "paired_accuracy", "convergence")},
            "figures": {name: directory / f"{name}.png" for name in ("fold_scores", "learning_curves")}}


def load_cv_report(config_path: str | Path = FROZEN_CONFIG) -> dict[str, Any]:
    """Read/check the completed comparison; no training or setup side effects."""
    config_path = REPO_ROOT / config_path
    config = _read(config_path)
    pointer = REPO_ROOT / config["paths"]["output_dir"] / "manifests/cross_validation.json"
    if not pointer.is_file():
        raise FileNotFoundError("Run python -m part1.snli_experiments before inspecting section 7.")
    report = _load_report(Path(_read(pointer)["manifest_path"]))
    if report["manifest"]["specification"]["configuration_sha256"] != _sha256(config_path):
        raise ValueError("Saved CV results use a different configuration.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--max-fits", type=int, help="Bound new fits; rerun without this limit to finish.")
    args = parser.parse_args()
    report = run_cross_validation(args.config, max_fits=args.max_fits, verbose=True)
    print("\nCV status:", report["manifest"]["status"])
    print("Manifest:", report["manifest_path"])
    if report["manifest"]["status"] == "complete":
        print(report["tables"]["summary"].to_string(index=False))
