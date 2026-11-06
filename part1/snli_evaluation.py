"""Freeze leakage-aware SNLI pools and repeated grouped CV assignments.

Run ``python -m part1.snli_evaluation`` after Step 2. No preprocessing or
classifier is fitted here. Test premises are used only for structural exclusion;
test labels and hypotheses are not used to construct the protocol.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from .project_setup import DEFAULT_CONFIG, prepare_workspace
from .snli_data import CLEANING_VERSION, ID_COLUMNS, LABEL_NAMES, SPLITS


PROTOCOL_VERSION = "v1"
POOL_FLAGS = ("cv_pool", "pilot_validation", "final_refit", "final_test")
ROW_COLUMNS = ["pool_index", "clean_index", *ID_COLUMNS, "premise_group", "label"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_complete_groups(frame: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    """Sample a label-independent group prefix, preserving original row order.

    Include the last complete group needed to reach the target. The actual row
    count can exceed the target by less than the last group's size.
    """
    if not isinstance(target_rows, int) or isinstance(target_rows, bool) or not 1 <= target_rows <= len(frame):
        raise ValueError("Group-sampling target must be between one and the eligible row count.")
    sizes = frame.groupby("premise_group", sort=True).size()
    shuffled = sizes.iloc[np.random.default_rng(seed).permutation(len(sizes))]
    count = int(np.searchsorted(shuffled.cumsum().to_numpy(), target_rows)) + 1
    return frame.loc[frame["premise_group"].isin(shuffled.index[:count])].copy().reset_index(drop=True)


def apply_cv_group_budget(pools: dict[str, pd.DataFrame], settings: dict[str, Any]) -> None:
    """Derive a separate budgeted CV selection without changing holdout pools."""
    selected = select_complete_groups(pools["cv_pool"], settings["target_rows"], settings["seed"])
    kept = set(selected["example_id"])
    excluded = pools["cv_pool"].loc[~pools["cv_pool"]["example_id"].isin(kept)].copy()
    excluded = excluded[["clean_index", *ID_COLUMNS, "premise_group"]]
    excluded["pool"], excluded["reason"] = "cv_pool", "not_selected_by_fixed_group_budget"
    pools["excluded_rows"] = pd.concat([pools["excluded_rows"], excluded], ignore_index=True)
    selected["pool_index"] = np.arange(len(selected), dtype=np.int64)
    pools["cv_pool"] = selected[ROW_COLUMNS]
    pools["membership"]["cv_pool"] = pools["membership"]["example_id"].isin(kept)
    mask = pools["pool_summary"]["pool"].eq("cv_pool")
    pools["pool_summary"].loc[mask, "rows"] = len(selected)
    pools["pool_summary"].loc[mask, "premise_groups"] = selected["premise_group"].nunique()


def make_development_pools(cleaned: DatasetDict) -> dict[str, pd.DataFrame]:
    """Preserve reference splits and derive selections with holdout priority.

    CV uses training rows whose exact premise is absent from both official
    holdouts. Pilot validation excludes test premises. Final refitting may use
    original train + validation, excluding test premises from both. Selection
    does not inspect validation/test labels, or any hypotheses.
    """
    if set(cleaned) != set(SPLITS):
        raise ValueError("Expected cleaned train, validation, and test splits.")
    frames = {}
    for split in SPLITS:
        frame = cleaned[split].select_columns([*ID_COLUMNS, "premise"]).to_pandas()
        if frame.empty or frame.isna().any().any():
            raise ValueError(f"{split}: expected nonempty, cleaned source metadata.")
        if not frame["source_split"].eq(split).all():
            raise ValueError(f"{split}: inconsistent original split identifiers.")
        expected_ids = split + ":" + frame["source_row"].astype(str)
        if not frame["example_id"].equals(expected_ids.rename("example_id")):
            raise ValueError(f"{split}: inconsistent original row identifiers.")
        frame.insert(0, "clean_index", np.arange(len(frame), dtype=np.int64))
        frames[split] = frame

    membership = pd.concat(frames.values(), ignore_index=True)
    if membership["example_id"].duplicated().any():
        raise ValueError("Source example IDs must be globally unique.")
    premise_ids = {
        text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for text in membership["premise"].unique()
    }
    membership["premise_group"] = membership["premise"].map(premise_ids)
    in_validation = membership["premise"].isin(frames["validation"]["premise"])
    in_test = membership["premise"].isin(frames["test"]["premise"])
    is_train = membership["source_split"].eq("train")
    is_validation = membership["source_split"].eq("validation")
    membership["cv_pool"] = is_train & ~in_validation & ~in_test
    membership["pilot_validation"] = is_validation & ~in_test
    membership["final_refit"] = (is_train | is_validation) & ~in_test
    membership["final_test"] = membership["source_split"].eq("test")

    exclusions = []
    for pool, candidates in (
        ("cv_pool", is_train), ("pilot_validation", is_validation),
        ("final_refit", is_train | is_validation),
    ):
        mask = candidates & ~membership[pool]
        excluded = membership.loc[mask, ["clean_index", *ID_COLUMNS, "premise_group"]].copy()
        excluded["pool"] = pool
        excluded["reason"] = [
            ";".join(
                reason for applies, reason in (
                    (pool == "cv_pool" and in_validation.loc[row], "premise_in_official_validation"),
                    (in_test.loc[row], "premise_in_official_test"),
                ) if applies
            )
            for row in excluded.index
        ]
        exclusions.append(excluded)

    # Gold labels are only read for the training rows used in stratification.
    train_labels = np.asarray(cleaned["train"]["label"])
    cv_pool = membership.loc[membership["cv_pool"]].copy().reset_index(drop=True)
    cv_pool.insert(0, "pool_index", np.arange(len(cv_pool), dtype=np.int64))
    cv_pool["label"] = train_labels[cv_pool["clean_index"].to_numpy()]
    if not cv_pool["label"].isin(range(len(LABEL_NAMES))).all():
        raise ValueError("The CV pool must contain only cleaned labels 0, 1, 2.")

    summary = []
    for pool in POOL_FLAGS:
        selected = membership.loc[membership[pool]]
        summary.append({
            "pool": pool, "rows": len(selected),
            "premise_groups": selected["premise_group"].nunique(),
        })
    groups = {pool: set(membership.loc[membership[pool], "premise_group"]) for pool in POOL_FLAGS}
    if groups["cv_pool"] & groups["pilot_validation"] or groups["final_refit"] & groups["final_test"]:
        raise ValueError("Development/holdout premise overlap remains after exclusion.")
    return {
        "cv_pool": cv_pool[ROW_COLUMNS],
        "membership": membership.drop(columns="premise"),
        "excluded_rows": pd.concat(exclusions, ignore_index=True),
        "pool_summary": pd.DataFrame(summary),
    }


def validate_folds(
    assignments: pd.DataFrame, cv_pool: pd.DataFrame, num_folds: int, seeds: list[int],
) -> pd.DataFrame:
    """Check row alignment, coverage, class presence, and group separation."""
    if assignments["example_id"].duplicated().any():
        raise ValueError("Each CV example must have exactly one assignment row.")
    if not assignments[ROW_COLUMNS].equals(cv_pool[ROW_COLUMNS]):
        raise ValueError("Saved folds do not match the CV pool's rows/order/labels/groups.")
    expected_columns = ROW_COLUMNS + [f"fold_seed_{seed}" for seed in seeds]
    if assignments.columns.tolist() != expected_columns:
        raise ValueError("Unexpected fold columns or repetition order.")

    summaries = []
    labels = assignments["label"].to_numpy()
    for repetition, seed in enumerate(seeds, start=1):
        column = f"fold_seed_{seed}"
        folds = assignments[column]
        if not pd.api.types.is_integer_dtype(folds) or not folds.isin(range(num_folds)).all():
            raise ValueError(f"Seed {seed}: every row needs one valid integer held-out fold.")
        if assignments.groupby("premise_group")[column].nunique().gt(1).any():
            raise ValueError(f"Seed {seed}: a premise group crosses held-out folds.")
        coverage = np.zeros(len(assignments), dtype=np.int8)
        for fold in range(num_folds):
            heldout = folds.to_numpy() == fold
            coverage += heldout
            train_counts = np.bincount(labels[~heldout], minlength=len(LABEL_NAMES))
            heldout_counts = np.bincount(labels[heldout], minlength=len(LABEL_NAMES))
            if not train_counts.all() or not heldout_counts.all():
                raise ValueError(f"Seed {seed}, fold {fold}: a partition is empty or lacks a class.")
            row = {
                "repetition": repetition, "seed": seed, "fold": fold,
                "train_rows": int((~heldout).sum()), "heldout_rows": int(heldout.sum()),
                "train_groups": assignments.loc[~heldout, "premise_group"].nunique(),
                "heldout_groups": assignments.loc[heldout, "premise_group"].nunique(),
                "shared_premise_groups": 0,
            }
            for label, name in enumerate(LABEL_NAMES):
                row[f"train_{name}"] = int(train_counts[label])
                row[f"heldout_{name}"] = int(heldout_counts[label])
                row[f"heldout_{name}_proportion"] = float(heldout_counts[label] / heldout.sum())
            summaries.append(row)
        if not np.all(coverage == 1):
            raise ValueError(f"Seed {seed}: held-out coverage is not exactly once per row.")
    return pd.DataFrame(summaries)


def build_grouped_folds(
    cv_pool: pd.DataFrame, num_folds: int, seeds: list[int], *, verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign complete exact-premise groups to approximately stratified folds."""
    if num_folds < 2 or cv_pool["premise_group"].nunique() < num_folds:
        raise ValueError("CV needs at least two folds and at least one group per fold.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide a nonempty list of distinct CV split seeds.")
    assignments = cv_pool[ROW_COLUMNS].copy()
    for seed in seeds:
        if verbose:
            print(f"Preparing {num_folds} grouped folds for seed {seed}...", flush=True)
        splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold_ids = np.full(len(cv_pool), -1, dtype=np.int16)
        for fold, (_, heldout) in enumerate(splitter.split(
            np.zeros(len(cv_pool)), cv_pool["label"], groups=cv_pool["premise_group"],
        )):
            fold_ids[heldout] = fold
        assignments[f"fold_seed_{seed}"] = fold_ids
    return assignments, validate_folds(assignments, cv_pool, num_folds, seeds)


def prepare_evaluation_protocol(
    config_path: str | Path = DEFAULT_CONFIG, *, verbose: bool = False,
) -> dict[str, Any]:
    """Save once and reuse checked fold assignments for the same data/design.

    A changed pool, seed, splitter version, or evaluation policy gets a new
    protocol directory. Training hyperparameters do not change the folds.
    Checksums reject modified/incomplete saved artifacts instead of overwriting.
    """
    workspace = prepare_workspace(config_path)
    config, paths = workspace["config"], workspace["paths"]
    prepared_dir = paths["data"] / "snli" / config["dataset"]["revision"] / f"cleaned-{CLEANING_VERSION}"
    if not prepared_dir.is_dir():
        raise FileNotFoundError(f"Run Step 2 first: python -m part1.snli_data --config {config_path}")
    cleaned = load_from_disk(str(prepared_dir))
    if not isinstance(cleaned, DatasetDict):
        raise ValueError("Expected the saved cleaned DatasetDict from Step 2.")
    pools = make_development_pools(cleaned)
    cv_config = config["cross_validation"]
    if cv_config["group_by"] != "premise" or cv_config["splitter"] != "StratifiedGroupKFold" or not cv_config["shuffle"]:
        raise ValueError("This protocol requires shuffled, exact-premise StratifiedGroupKFold.")
    if cv_config["pool"] == "fixed_group_subset":
        apply_cv_group_budget(pools, cv_config["subset"])
    elif cv_config["pool"] != "all_eligible_training_rows":
        raise ValueError("Expected all_eligible_training_rows or fixed_group_subset.")
    seeds, num_folds = config["seeds"]["cv_splits"], cv_config["num_folds"]
    policy = config["evaluation"]["pool_policy"]
    if policy != "test_priority_exclude_train_overlap_with_both_holdouts_v1":
        raise ValueError(f"Unsupported pool policy: {policy}")
    specification = {
        "protocol_version": PROTOCOL_VERSION, "dataset": config["dataset"],
        "cleaning_version": CLEANING_VERSION,
        "cleaned_fingerprints": {split: cleaned[split]._fingerprint for split in SPLITS},
        "cross_validation": cv_config, "split_seeds": seeds,
        "evaluation": config["evaluation"], "sklearn_version": sklearn.__version__,
    }
    protocol_id = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:16]
    protocol_dir = paths["folds"] / protocol_id
    manifest_path = protocol_dir / "protocol.json"
    reused = protocol_dir.exists()
    if reused:
        if not manifest_path.is_file():
            raise ValueError(f"Incomplete protocol directory: {protocol_dir}. No saved folds were overwritten.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["specification"] != specification:
            raise ValueError("Saved protocol specification does not match the current data/design.")
        for name, digest in manifest["artifact_sha256"].items():
            path = protocol_dir / name
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"Saved evaluation artifact changed or is missing: {path}")
        assignments = pd.read_parquet(protocol_dir / "fold_assignments.parquet")
        fold_summary = validate_folds(assignments, pools["cv_pool"], num_folds, seeds)
    else:
        assignments, fold_summary = build_grouped_folds(pools["cv_pool"], num_folds, seeds, verbose=verbose)
        protocol_dir.mkdir()
        assignments.to_parquet(protocol_dir / "fold_assignments.parquet", index=False)
        pools["membership"].to_parquet(protocol_dir / "pool_membership.parquet", index=False)
        for name, frame in (
            ("excluded_rows", pools["excluded_rows"]), ("pool_summary", pools["pool_summary"]),
            ("fold_summary", fold_summary),
        ):
            frame.to_csv(protocol_dir / f"{name}.csv", index=False)
        manifest = {
            "stage": "evaluation_design", "protocol_id": protocol_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "specification": specification,
            "creation_config_sha256": workspace["manifest"]["config_sha256"],
            "repository": workspace["manifest"]["repository"],
            "implementation_sha256": _sha256(Path(__file__)),
            "prepared_dataset": str(prepared_dir),
            "pools": pools["pool_summary"].to_dict(orient="records"),
            "checks": {
                "unique_example_ids": True, "original_row_order_preserved": True,
                "premise_disjoint_cv_partitions": True,
                "one_heldout_assignment_per_row_per_repetition": True,
                "all_classes_present_in_each_partition": True,
                "cv_disjoint_from_official_holdout_premises": True,
                "pilot_validation_disjoint_from_test_premises": True,
                "final_refit_disjoint_from_test_premises": True,
                "cleaned_official_reference_splits_unchanged": True,
            },
            "models_trained": False,
            "test_usage": "Premises and source IDs only for exclusions; no test labels/hypotheses or model scores.",
            "selection_policy": {
                "pilot": "Tune limited settings using eligible training groups and pilot validation; fix before reported CV.",
                "reported_cv": "Fit tokenizer/features/model on each outer training partition only; use fixed epochs from pilot.",
                "pairing": "All systems/baselines reuse the saved pool_index order and fold_seed columns.",
                "budget_change": "If the pilot requires fewer complete groups, save a separately identified subset/protocol before comparisons.",
                "final_refit": "Freeze selected system; use original cleaned train + validation excluding test premises; test once.",
            },
            "artifact_sha256": {
                path.name: _sha256(path) for path in sorted(protocol_dir.iterdir()) if path.is_file()
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # A small pointer is refreshed; the fold assignments and protocol are immutable.
    latest_path = paths["manifests"] / "evaluation_design.json"
    latest_path.write_text(json.dumps({
        "protocol_id": protocol_id, "protocol_manifest": str(manifest_path),
        "current_config_sha256": workspace["manifest"]["config_sha256"],
    }, indent=2) + "\n", encoding="utf-8")
    return {
        **pools, "assignments": assignments, "fold_summary": fold_summary,
        "manifest": manifest, "manifest_path": manifest_path,
        "protocol_dir": protocol_dir, "reused_folds": reused,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    protocol = prepare_evaluation_protocol(arguments.config, verbose=True)
    print(protocol["pool_summary"].to_string(index=False))
    print("\nFold sizes:")
    print(protocol["fold_summary"][["seed", "fold", "train_rows", "heldout_rows", "shared_premise_groups"]].to_string(index=False))
    print("\nReused saved folds:", protocol["reused_folds"])
    print("Protocol:", protocol["manifest_path"])
