"""Offline checks for selection boundaries, grouped folds, and saved reuse."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
import numpy as np
import pandas as pd

from part1.project_setup import DEFAULT_CONFIG
from part1.snli_data import CLEANING_VERSION, LABEL_NAMES, clean_split
from part1.snli_evaluation import (
    build_grouped_folds, make_development_pools,
    prepare_evaluation_protocol, validate_folds,
)


def cleaned_split(split, premises, labels=None):
    if labels is None:
        labels = [i % 3 for i in range(len(premises))]
    source = Dataset.from_dict({
        "premise": premises,
        "hypothesis": [f"hypothesis {i}" for i in range(len(premises))],
        "label": labels,
    }, features=Features({
        "premise": Value("string"), "hypothesis": Value("string"),
        "label": ClassLabel(names=list(LABEL_NAMES)),
    }))
    return clean_split(source, split)[0]


def example_splits():
    return DatasetDict({
        "train": cleaned_split("train", [f"premise {i}" for i in range(18) for _ in range(3)]),
        "validation": cleaned_split("validation", ["validation only"] * 3),
        "test": cleaned_split("test", ["test only"] * 3),
    })


class DevelopmentPoolTests(unittest.TestCase):
    def test_holdout_priority_and_source_order_without_test_targets(self):
        cleaned = DatasetDict({
            "train": cleaned_split("train", [
                "kept", "validation shared", "test shared", "both shared", "kept", "kept ",
            ]),
            "validation": cleaned_split("validation", [
                "validation shared", "both shared", "test and validation", "validation only",
            ]),
            "test": cleaned_split("test", [
                "test shared", "both shared", "test and validation", "test only",
            ]),
        })
        # Selection must work without reading holdout labels or hypotheses.
        for split in ("validation", "test"):
            cleaned[split] = cleaned[split].remove_columns(["label", "hypothesis"])
        report = make_development_pools(cleaned)
        cv = report["cv_pool"]
        self.assertEqual(cv["example_id"].tolist(), ["train:0", "train:4", "train:5"])
        self.assertEqual(cv["pool_index"].tolist(), [0, 1, 2])
        self.assertEqual(cv["clean_index"].tolist(), [0, 4, 5])
        self.assertEqual(cv.iloc[0]["premise_group"], cv.iloc[1]["premise_group"])
        self.assertNotEqual(cv.iloc[0]["premise_group"], cv.iloc[2]["premise_group"])
        membership = report["membership"].set_index("example_id")
        self.assertTrue(membership.loc["validation:0", "pilot_validation"])
        self.assertFalse(membership.loc["validation:1", "pilot_validation"])
        # Train/validation overlap may return for final refitting, never test overlap.
        self.assertTrue(membership.loc["train:1", "final_refit"])
        self.assertFalse(membership.loc["train:2", "final_refit"])
        self.assertEqual(int(membership["final_test"].sum()), 4)
        exclusions = report["excluded_rows"]
        reason = exclusions.loc[
            (exclusions["example_id"] == "train:3") & (exclusions["pool"] == "cv_pool"), "reason",
        ].item()
        self.assertEqual(reason, "premise_in_official_validation;premise_in_official_test")
        self.assertNotIn("label", membership.columns)

    def test_reject_duplicate_or_wrong_source_ids(self):
        cleaned = example_splits()
        frame = cleaned["train"].to_pandas()
        frame.loc[1, "example_id"] = frame.loc[0, "example_id"]
        cleaned["train"] = Dataset.from_pandas(frame, preserve_index=False)
        with self.assertRaisesRegex(ValueError, "row identifiers"):
            make_development_pools(cleaned)


class GroupedFoldTests(unittest.TestCase):
    def setUp(self):
        self.pool = make_development_pools(example_splits())["cv_pool"]
        self.seeds = [0, 3]

    def test_reproducible_complete_grouped_folds_and_class_counts(self):
        first, summary = build_grouped_folds(self.pool, 3, self.seeds)
        second, _ = build_grouped_folds(self.pool, 3, self.seeds)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(summary), 6)
        self.assertTrue(summary["shared_premise_groups"].eq(0).all())
        for seed in self.seeds:
            column = f"fold_seed_{seed}"
            self.assertEqual(first.groupby("premise_group")[column].nunique().max(), 1)
            counts = sum((first[column].to_numpy() == fold).astype(int) for fold in range(3))
            np.testing.assert_array_equal(counts, np.ones(len(first), dtype=int))
            self.assertEqual(summary.loc[summary["seed"] == seed, "heldout_rows"].sum(), len(self.pool))
        self.assertEqual(first["example_id"].tolist(), self.pool["example_id"].tolist())

    def test_reject_group_leakage_and_missing_or_reordered_rows(self):
        assignments, _ = build_grouped_folds(self.pool, 3, self.seeds)
        broken = assignments.copy()
        broken.loc[0, "fold_seed_0"] = (int(broken.loc[0, "fold_seed_0"]) + 1) % 3
        with self.assertRaisesRegex(ValueError, "premise group"):
            validate_folds(broken, self.pool, 3, self.seeds)
        for broken in (assignments.iloc[1:], assignments.iloc[::-1].reset_index(drop=True)):
            with self.assertRaisesRegex(ValueError, "rows/order"):
                validate_folds(broken, self.pool, 3, self.seeds)

    def test_reject_unassigned_rows_and_invalid_design(self):
        assignments, _ = build_grouped_folds(self.pool, 3, self.seeds)
        assignments.loc[0, "fold_seed_0"] = -1
        with self.assertRaisesRegex(ValueError, "valid integer"):
            validate_folds(assignments, self.pool, 3, self.seeds)
        for folds, seeds in ((1, [0]), (100, [0]), (3, []), (3, [0, 0])):
            with self.assertRaises(ValueError):
                build_grouped_folds(self.pool, folds, seeds)


class SavedProtocolTests(unittest.TestCase):
    def test_reuse_is_immutable_and_detects_changed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(json.loads(DEFAULT_CONFIG.read_text()))
            config["paths"] = {"data_dir": str(root / "data"), "output_dir": str(root / "results")}
            config["cross_validation"]["num_folds"] = 3
            config["seeds"]["cv_splits"] = [0, 3]
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            prepared_dir = root / "data/snli" / config["dataset"]["revision"] / f"cleaned-{CLEANING_VERSION}"
            example_splits().save_to_disk(str(prepared_dir))

            first = prepare_evaluation_protocol(config_path)
            assignments_path = first["protocol_dir"] / "fold_assignments.parquet"
            initial_mtime = assignments_path.stat().st_mtime_ns
            # Model tuning must not silently regenerate CV folds.
            config["neural_training"]["learning_rate"] = 0.002
            config_path.write_text(json.dumps(config))
            second = prepare_evaluation_protocol(config_path)
            self.assertFalse(first["reused_folds"])
            self.assertTrue(second["reused_folds"])
            self.assertEqual(first["protocol_dir"], second["protocol_dir"])
            self.assertEqual(initial_mtime, assignments_path.stat().st_mtime_ns)
            pd.testing.assert_frame_equal(first["assignments"], second["assignments"])
            pd.testing.assert_frame_equal(first["fold_summary"], second["fold_summary"])

            config["seeds"]["cv_splits"] = [0, 5]
            config_path.write_text(json.dumps(config))
            changed = prepare_evaluation_protocol(config_path)
            self.assertFalse(changed["reused_folds"])
            self.assertNotEqual(first["protocol_dir"], changed["protocol_dir"])
            self.assertEqual(initial_mtime, assignments_path.stat().st_mtime_ns)
            config["seeds"]["cv_splits"] = [0, 3]
            config_path.write_text(json.dumps(config))

            with assignments_path.open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                prepare_evaluation_protocol(config_path)


if __name__ == "__main__":
    unittest.main()
