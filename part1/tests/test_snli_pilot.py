"""Offline pilot selection, observers, group budgets, and end-to-end checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import DatasetDict
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from part1.project_setup import DEFAULT_CONFIG
from part1.snli_data import CLEANING_VERSION
from part1.snli_evaluation import (
    apply_cv_group_budget, make_development_pools, prepare_evaluation_protocol,
    select_complete_groups,
)
from part1.snli_models import fit_classifier
from part1.snli_pilot import (
    DEFAULT_PILOT, load_pilot_report, project_comparison_budget, run_pilot,
    select_common_epoch,
)
from part1.tests.test_snli_evaluation import cleaned_split, example_splits


class GroupBudgetTests(unittest.TestCase):
    def test_complete_groups_are_seeded_nested_and_keep_row_order(self):
        frame = pd.DataFrame({
            "premise_group": ["a", "a", "b", "b", "b", "c", "d", "d"],
            "original_row": list(range(8)),
        })
        small = select_complete_groups(frame, 4, 0)
        large = select_complete_groups(frame, 7, 0)
        pd.testing.assert_frame_equal(small, select_complete_groups(frame, 4, 0))
        self.assertTrue(set(small["premise_group"]).issubset(large["premise_group"]))
        self.assertEqual(small["original_row"].tolist(), sorted(small["original_row"]))
        self.assertGreaterEqual(len(small), 4)
        self.assertLess(len(small), 7)
        for group in small["premise_group"].unique():
            self.assertEqual((small["premise_group"] == group).sum(), (frame["premise_group"] == group).sum())
        with self.assertRaises(ValueError):
            select_complete_groups(frame, 9, 0)

    def test_cv_budget_changes_only_cv_membership_and_audits_exclusions(self):
        original = make_development_pools(example_splits())
        budgeted = {name: frame.copy(deep=True) for name, frame in original.items()}
        apply_cv_group_budget(budgeted, {"target_rows": 20, "seed": 0})
        self.assertEqual(len(budgeted["cv_pool"]), 21)
        np.testing.assert_array_equal(budgeted["cv_pool"]["pool_index"], np.arange(21))
        for flag in ("pilot_validation", "final_refit", "final_test"):
            np.testing.assert_array_equal(original["membership"][flag], budgeted["membership"][flag])
        excluded = budgeted["excluded_rows"].query("reason == 'not_selected_by_fixed_group_budget'")
        self.assertEqual(len(excluded) + len(budgeted["cv_pool"]), len(original["cv_pool"]))
        self.assertFalse(set(excluded["example_id"]) & set(budgeted["cv_pool"]["example_id"]))


class PilotSelectionTests(unittest.TestCase):
    def test_common_epoch_prefers_earlier_practically_equivalent_schedule(self):
        rows = [
            {"epoch": epoch, "representation": representation, "classifier": kind,
             "validation_accuracy": score, "validation_nll": 1 - score}
            for epoch, score in ((3, .60), (5, .647), (8, .65))
            for representation in ("indicators", "unigram_counts", "unigram_bigram_counts")
            for kind in ("mlp_one_hidden", "mlp_two_hidden")
        ]
        selected, table = select_common_epoch(pd.DataFrame(rows), [3, 5, 8], .005)
        self.assertEqual(selected, 5)
        self.assertEqual(table.loc[table["selected"], "epoch"].tolist(), [5])
        with self.assertRaises(ValueError):
            select_common_epoch(pd.DataFrame(rows).iloc[1:], [3, 5, 8], .005)

    def test_budget_projection_is_explicit_and_chooses_largest_feasible_pool(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        policy = json.loads(DEFAULT_PILOT.read_text())
        policy.update({"comparison_row_candidates": [100, 200], "include_full_pool_candidate": True,
                       "primary_matrix_budget_seconds": 80, "runtime_projection_safety_factor": 2})
        selected, table = project_comparison_budget(1., 0., 100, 1000, config, policy)
        self.assertEqual(selected, 200)
        self.assertEqual(table.loc[table["target_rows"] == 100, "estimated_primary_seconds"].item(), 12.)
        self.assertEqual(table.loc[table["target_rows"] == 1000, "conservative_primary_seconds"].item(), 240.)

    def test_observer_does_not_change_epochs_or_training_trajectory(self):
        config = deepcopy(json.loads(DEFAULT_CONFIG.read_text()))
        config["neural_training"].update({"num_threads": 1, "batch_size": 6})
        config["neural_training"]["hidden_dims"]["mlp_one_hidden"] = [8]
        X = sparse.csr_matrix(np.tile(np.eye(3, dtype=np.float32), (2, 1)))
        y = np.tile(np.arange(3), 2)
        seen = []

        def observe(model):
            seen.append((len(model.history), model.predict_proba(X)))
            return "stop"  # The trainer ignores observer return values.

        threads = torch.get_num_threads()
        observed = fit_classifier("mlp_one_hidden", X, y, config, num_epochs=3, epoch_observer=observe)
        control = fit_classifier("mlp_one_hidden", X, y, config, num_epochs=3)
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual([epoch for epoch, _ in seen], [1, 2, 3])
        self.assertEqual(observed.fit_info["epochs_completed"], 3)
        np.testing.assert_array_equal(observed.predict_proba(X), control.predict_proba(X))


class PilotIntegrationTests(unittest.TestCase):
    def test_pilot_records_predictions_freezes_separate_folds_and_reuses_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(json.loads(DEFAULT_CONFIG.read_text()))
            config["paths"] = {"data_dir": str(root / "data"), "output_dir": str(root / "results")}
            config["cross_validation"]["num_folds"] = 3
            config["seeds"]["cv_splits"] = [0, 3]
            config["tokenizer"]["vocab_size"] = 64
            config["bigram_features"]["max_features"] = 10
            config["neural_training"]["hidden_dims"] = {"mlp_one_hidden": [8], "mlp_two_hidden": [8, 4]}
            config_path, policy_path, frozen_path = root / "base.json", root / "policy.json", root / "frozen.json"
            config_path.write_text(json.dumps(config))
            policy = deepcopy(json.loads(DEFAULT_PILOT.read_text()))
            policy.update({"training_target_rows": 12, "validation_target_rows": 6,
                           "epoch_candidates": [1, 2], "comparison_row_candidates": [30],
                           "include_full_pool_candidate": False})
            policy_path.write_text(json.dumps(policy))
            dataset = DatasetDict({
                "train": cleaned_split("train", [f"training {i}" for i in range(18) for _ in range(3)]),
                "validation": cleaned_split("validation", [f"validation {i}" for i in range(6) for _ in range(3)]),
                "test": cleaned_split("test", ["test only"] * 3),
            })
            prepared = root / "data/snli" / config["dataset"]["revision"] / f"cleaned-{CLEANING_VERSION}"
            dataset.save_to_disk(str(prepared))
            original = prepare_evaluation_protocol(config_path)
            report = run_pilot(config_path, policy_path, frozen_config_path=frozen_path)
            self.assertTrue(all(report["manifest"]["checks"].values()))
            self.assertEqual(report["manifest"]["training_rows"], 12)
            self.assertEqual(report["manifest"]["validation_rows"], 6)
            self.assertEqual(len(report["tables"]["metrics"]), 11)
            self.assertEqual(report["manifest"]["comparison_actual_rows"], 30)
            self.assertIn(report["manifest"]["selected_epochs"], [1, 2])
            self.assertEqual(report["frozen_config"]["cross_validation"]["pool"], "fixed_group_subset")
            final = prepare_evaluation_protocol(frozen_path)
            self.assertNotEqual(original["manifest_path"], final["manifest_path"])
            self.assertTrue(original["manifest_path"].is_file())
            for path in report["pilot_dir"].glob("selected_models/*/predictions.parquet"):
                predictions = pd.read_parquet(path)
                self.assertTrue(predictions["source_split"].eq("validation").all())
                self.assertTrue(predictions["stage"].eq("pilot_validation").all())
                self.assertEqual(len(predictions), 6)
            with patch("part1.snli_pilot.fit_classifier", side_effect=AssertionError("Must reuse finished pilot")):
                cached = run_pilot(config_path, policy_path, frozen_config_path=frozen_path)
                loaded = load_pilot_report(config_path)
            self.assertEqual(cached["manifest_path"], report["manifest_path"])
            pd.testing.assert_frame_equal(loaded["tables"]["metrics"], report["tables"]["metrics"])
            corrupted = report["pilot_dir"] / "metrics.csv"
            with corrupted.open("a") as stream:
                stream.write("modified")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_pilot_report(config_path)


if __name__ == "__main__":
    unittest.main()
