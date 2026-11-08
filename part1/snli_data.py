"""Load, clean, and inspect the pinned SNLI dataset for Project 1.

Run ``python -m part1.snli_data`` from the repository root. This performs a
structural dataset audit; it does not fit a tokenizer or classifier.
"""

import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from datasets import load_dataset
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from .project_setup import DEFAULT_CONFIG, prepare_workspace


SPLITS = ("train", "validation", "test")
LABEL_NAMES = ("entailment", "neutral", "contradiction")
TEXT_COLUMNS = ("premise", "hypothesis")
ID_COLUMNS = ("example_id", "source_split", "source_row")
CLEANING_VERSION = "v1"


def validate_label_mapping(dataset: Dataset) -> None:
    """Reject an unexpected schema rather than silently changing label meaning."""
    required = {*TEXT_COLUMNS, "label"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"SNLI requires columns {sorted(required)}.")
    label = dataset.features.get("label")
    if not isinstance(label, ClassLabel) or label.names != list(LABEL_NAMES):
        raise ValueError(f"Unexpected SNLI label mapping: {label!r}")


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def clean_split(dataset: Dataset, split: str) -> tuple[Dataset, pd.DataFrame]:
    """Filter unusable rows, retaining original text, labels, order, and row IDs.

    Invalid labels take precedence in the removal audit when a row also has
    empty text, so exclusion reasons are mutually exclusive. Duplicates remain.
    """
    validate_label_mapping(dataset)
    frame = dataset.to_pandas()
    frame["source_row"] = np.arange(len(frame), dtype=np.int64)
    frame["source_split"] = split
    frame["example_id"] = [f"{split}:{row}" for row in frame["source_row"]]

    valid_label = frame["label"].isin(range(len(LABEL_NAMES)))
    valid_text = frame["premise"].map(_has_text) & frame["hypothesis"].map(_has_text)
    keep = valid_label & valid_text
    removed = frame.loc[~keep, [*ID_COLUMNS, "label"]].copy()
    removed["reason"] = np.where(
        ~valid_label.loc[~keep], "invalid_label", "missing_or_empty_text"
    )

    features = Features({
        **dataset.features,
        "source_row": Value("int64"),
        "source_split": Value("string"),
        "example_id": Value("string"),
    })
    cleaned = Dataset.from_pandas(frame.loc[keep], features=features, preserve_index=False)
    return cleaned, removed.reset_index(drop=True)


def build_data_report(
    raw: DatasetDict, cleaned: DatasetDict, removed: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Summarize labels, lengths, duplicates, and exact cross-split overlap.

    Lengths use Python whitespace splitting for exploration, not a learned BPE
    tokenizer. Overlap uses exact original strings, without normalization.
    """
    frames = {split: cleaned[split].to_pandas() for split in SPLITS}
    summaries, labels, lengths, duplicate_groups = [], [], [], []
    premise_sets: dict[str, set[str]] = {}
    pair_sets: dict[str, set[tuple[str, str]]] = {}

    for split, frame in frames.items():
        source = raw[split].to_pandas()
        exclusions = removed.loc[removed["source_split"] == split, "reason"]
        premise_counts = frame["premise"].value_counts()
        repeated_pairs = frame.loc[frame.duplicated(list(TEXT_COLUMNS), keep=False)]
        conflicts = 0
        for (premise, hypothesis), group in repeated_pairs.groupby(
            list(TEXT_COLUMNS), sort=False
        ):
            unique_labels = sorted(int(label) for label in group["label"].unique())
            conflicts += len(unique_labels) > 1
            duplicate_groups.append({
                "split": split, "premise": premise, "hypothesis": hypothesis,
                "rows": len(group), "distinct_labels": len(unique_labels),
                "labels": json.dumps(unique_labels),
                "example_ids": json.dumps(group["example_id"].tolist()),
            })
        summaries.append({
            "split": split,
            "raw_rows": len(source),
            "invalid_label_rows": int((exclusions == "invalid_label").sum()),
            "invalid_text_rows_after_label_filter": int(
                (exclusions == "missing_or_empty_text").sum()
            ),
            "raw_empty_or_missing_premise": int((~source["premise"].map(_has_text)).sum()),
            "raw_empty_or_missing_hypothesis": int((~source["hypothesis"].map(_has_text)).sum()),
            "clean_rows": len(frame),
            "unique_premises": frame["premise"].nunique(),
            "repeated_premise_groups": int((premise_counts > 1).sum()),
            "rows_in_repeated_premise_groups": int(premise_counts[premise_counts > 1].sum()),
            "largest_premise_group": int(premise_counts.max()) if len(frame) else 0,
            "extra_duplicate_pair_rows": int(frame.duplicated(list(TEXT_COLUMNS)).sum()),
            "conflicting_label_pair_groups": conflicts,
        })
        counts = frame["label"].value_counts().reindex(range(len(LABEL_NAMES)), fill_value=0)
        for label, name in enumerate(LABEL_NAMES):
            labels.append({
                "split": split, "label": label, "label_name": name,
                "count": int(counts[label]),
                "proportion": float(counts[label] / len(frame)) if len(frame) else 0.0,
            })

        per_row = frame[["example_id"]].copy()
        per_row["split"] = split
        for field in TEXT_COLUMNS:
            per_row[f"{field}_characters"] = frame[field].map(len)
            per_row[f"{field}_whitespace_tokens"] = frame[field].map(lambda text: len(text.split()))
        lengths.append(per_row)
        premise_sets[split] = set(frame["premise"])
        pair_sets[split] = set(zip(frame["premise"], frame["hypothesis"]))

    row_lengths = pd.concat(lengths, ignore_index=True)
    length_summary = []
    for split in SPLITS:
        split_lengths = row_lengths.loc[row_lengths["split"] == split]
        for field in TEXT_COLUMNS:
            for unit in ("characters", "whitespace_tokens"):
                values = split_lengths[f"{field}_{unit}"]
                length_summary.append({
                    "split": split, "field": field, "unit": unit,
                    "min": values.min(), "mean": values.mean(),
                    "median": values.median(), "p95": values.quantile(0.95),
                    "max": values.max(),
                })

    overlaps = []
    for left, right in combinations(SPLITS, 2):
        shared_premises = premise_sets[left] & premise_sets[right]
        shared_pairs = pair_sets[left] & pair_sets[right]
        overlaps.append({
            "left_split": left, "right_split": right,
            "shared_unique_premises": len(shared_premises),
            "left_rows_with_shared_premise": int(frames[left]["premise"].isin(shared_premises).sum()),
            "right_rows_with_shared_premise": int(frames[right]["premise"].isin(shared_premises).sum()),
            "shared_unique_pairs": len(shared_pairs),
        })

    examples = frames["train"].drop_duplicates("label").sort_values("label").copy()
    examples["label_name"] = examples["label"].map(dict(enumerate(LABEL_NAMES)))
    return {
        "split_summary": pd.DataFrame(summaries),
        "label_counts": pd.DataFrame(labels),
        "length_summary": pd.DataFrame(length_summary),
        "row_lengths": row_lengths,
        "split_overlap": pd.DataFrame(overlaps),
        "duplicate_pairs": pd.DataFrame(duplicate_groups, columns=[
            "split", "premise", "hypothesis", "rows", "distinct_labels", "labels", "example_ids",
        ]),
        "removed_rows": removed,
        "training_examples": examples[["example_id", "premise", "hypothesis", "label", "label_name"]],
    }


def plot_data_report(tables: dict[str, pd.DataFrame], figure_dir: Path) -> dict[str, Path]:
    """Save label proportions and full-range training-length distributions."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    proportions = tables["label_counts"].pivot(
        index="split", columns="label_name", values="proportion"
    ).reindex(index=list(SPLITS), columns=list(LABEL_NAMES))
    fig, ax = plt.subplots(figsize=(8, 4))
    proportions.plot.bar(ax=ax, rot=0, color=["#3478a5", "#db9c36", "#b45454"])
    ax.set(xlabel="Official split (cleaned)", ylabel="Class proportion", title="SNLI class balance")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    class_path = figure_dir / "snli_class_balance.png"
    fig.savefig(class_path, dpi=160)
    plt.close(fig)

    training_lengths = tables["row_lengths"].loc[tables["row_lengths"]["split"] == "train"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, field, color in zip(axes, TEXT_COLUMNS, ["#3478a5", "#db9c36"]):
        values = training_lengths[f"{field}_whitespace_tokens"]
        ax.hist(values, bins=40, color=color, edgecolor="white")
        ax.axvline(values.median(), color="#333333", linestyle="--", label="Median")
        ax.set(xlabel="Whitespace-separated tokens", ylabel="Training examples", title=field.title())
        ax.legend()
    fig.suptitle("SNLI training text lengths (exploratory units, not BPE)")
    fig.tight_layout()
    length_path = figure_dir / "snli_training_lengths.png"
    fig.savefig(length_path, dpi=160)
    plt.close(fig)
    return {"class_balance": class_path, "training_lengths": length_path}


def prepare_snli_report(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the source, save cleaned splits, and persist the structural audit."""
    workspace = prepare_workspace(config_path)
    config, paths = workspace["config"], workspace["paths"]
    source = config["dataset"]
    raw = load_dataset(
        source["repo_id"], revision=source["revision"],
        cache_dir=str(paths["data"] / "huggingface"),
    )
    if not isinstance(raw, DatasetDict) or set(raw) != set(SPLITS):
        raise ValueError("Expected the official train, validation, and test splits.")

    cleaned, audits = DatasetDict(), []
    for split in SPLITS:
        cleaned[split], audit = clean_split(raw[split], split)
        audits.append(audit)
    removed = pd.concat(audits, ignore_index=True)
    tables = build_data_report(raw, cleaned, removed)
    prepared_dir = paths["data"] / "snli" / source["revision"] / f"cleaned-{CLEANING_VERSION}"
    cleaned.save_to_disk(str(prepared_dir))

    report_dir = paths["metrics"] / "data_audit"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)
    figures = plot_data_report(tables, paths["figures"])
    manifest = {
        "stage": "data_inspection", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": source, "dataset_loaded": True,
        "config_sha256": workspace["manifest"]["config_sha256"],
        "repository": workspace["manifest"]["repository"],
        "label_mapping": dict(enumerate(LABEL_NAMES)),
        "cleaning_version": CLEANING_VERSION,
        "cleaning_policy": {
            "exclude": ["label outside {0, 1, 2}", "missing or whitespace-only sentence"],
            "retain": ["original split", "original text", "original gold label", "duplicate pairs"],
            "id_scheme": "official_split:original_zero_based_row",
            "overlap_matching": "exact strings, without normalization",
            "example_selection": "first retained training row of each class",
        },
        "splits": tables["split_summary"].to_dict(orient="records"),
        "source_fingerprints": {split: raw[split]._fingerprint for split in SPLITS},
        "cleaned_fingerprints": {split: cleaned[split]._fingerprint for split in SPLITS},
        "artifacts": {
            "prepared_dataset": str(prepared_dir), "tables": str(report_dir),
            "figures": {name: str(path) for name, path in figures.items()},
        },
        "test_usage": "Structural/count audit only; no predictions, scoring, or model selection.",
    }
    manifest_path = paths["manifests"] / "data_inspection.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "raw": raw, "cleaned": cleaned, "tables": tables, "figures": figures,
        "manifest": manifest, "manifest_path": manifest_path, "prepared_dir": prepared_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    report = prepare_snli_report(arguments.config)
    print(report["tables"]["split_summary"].to_string(index=False))
    print("\nExact split overlap:")
    print(report["tables"]["split_overlap"].to_string(index=False))
    print("\nData manifest:", report["manifest_path"])
