"""Offline paired-CV completion, leakage boundaries, resume and corruption checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from part1.project_setup import DEFAULT_CONFIG
from part1.snli_data import CLEANING_VERSION
from part1.snli_evaluation import ROW_COLUMNS, prepare_evaluation_protocol
from part1.snli_experiments import (
    PROBABILITY_COLUMNS, _prepare_features, load_cv_report, planned_systems,
    run_cross_validation, summarize_metrics, validate_predictions,
)
from part1.snli_models import fit_classifier
from part1.tests.test_snli_evaluation import example_splits


def small_config(root):
    config = deepcopy(json.loads(DEFAULT_CONFIG.read_text()))
    config["status"] = "frozen_after_pilot_before_reported_cv"
    config["paths"] = {"data_dir": str(root / "data"), "output_dir": str(root / "results")}
    config["cross_validation"]["num_folds"] = 3
    config["seeds"]["cv_splits"] = [0]
    config["tokenizer"]["vocab_size"] = 64
    config["bigram_features"]["max_features"] = 10
    config["neural_training"].update({"num_epochs": 1, "num_threads": 1})
    config["neural_training"]["hidden_dims"] = {"mlp_one_hidden": [8], "mlp_two_hidden": [8, 4]}
    return config


class CvDesignTests(unittest.TestCase):
    def test_requires_frozen_epochs_and_all_current_systems(self):
        config = json.loads(DEFAULT_CONFIG.read_text())
        with self.assertRaisesRegex(ValueError, "frozen"):
            planned_systems(config)
        config["status"] = "frozen_after_pilot_before_reported_cv"
        with self.assertRaisesRegex(ValueError, "epoch"):
            planned_systems(config)
        config["neural_training"]["num_epochs"] = 3
        systems = planned_systems(config)
        self.assertEqual(len(systems), 11)
        self.assertEqual(sum(s["primary"] for s in systems), 9)
        config["representations"][-1] = "unigram_bigram_counts"
        with self.assertRaisesRegex(ValueError, "representations"):
            planned_systems(config)

    def test_predictions_reject_wrong_order_gold_duplicates_and_decisions(self):
        dataset = example_splits()
        from part1.snli_evaluation import make_development_pools
        rows = make_development_pools(dataset)["cv_pool"][ROW_COLUMNS].head(6).copy()
        rows["partition"] = "heldout"
        predictions = rows.copy()
        predictions["stage"] = "outer_cv_heldout"
        predictions["prediction"] = rows["label"]
        probabilities = np.full((6, 3), .1)
        probabilities[np.arange(6), rows["label"]] = .8
        predictions[PROBABILITY_COLUMNS] = probabilities
        self.assertEqual(validate_predictions(predictions, rows)["accuracy"], 1.)
        for change in ("order", "label", "decision", "probability", "stage"):
            altered = predictions.copy()
            if change == "order":
                altered = altered.iloc[::-1]
            elif change == "label":
                altered.loc[0, "label"] = 2
            elif change == "decision":
                altered.loc[0, "prediction"] = 2
            elif change == "probability":
                altered.loc[0, PROBABILITY_COLUMNS[0]] = np.nan
            else:
                altered.loc[0, "stage"] = "pilot_validation"
            with self.assertRaises(ValueError, msg=change):
                validate_predictions(altered, rows)

    def test_incomplete_feature_writes_are_preserved_but_corruption_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = small_config(root)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            protocol = {"manifest": {"protocol_id": "example"}}
            cache = root / "results/features/example/seed_0_fold_0/0123456789abcdef"
            cache.mkdir(parents=True)
            (cache / "partial.npz").write_text("unfinished")
            incomplete = ValueError(f"Incomplete feature cache: {cache}. Existing files were not overwritten.")
            with patch("part1.snli_experiments.prepare_fold_features", side_effect=[incomplete, {"recovered": True}]):
                self.assertEqual(_prepare_features(config_path, protocol, 0, 0, False), {"recovered": True})
            self.assertFalse(cache.exists())
            self.assertEqual(len(list(cache.parent.glob("interrupted/*/partial.npz"))), 1)
            with patch("part1.snli_experiments.prepare_fold_features", side_effect=ValueError("Saved feature artifact changed")):
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    _prepare_features(config_path, protocol, 0, 0, False)


class CvIntegrationTests(unittest.TestCase):
    def test_bounded_failed_and_completed_runs_resume_and_keep_paired_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = small_config(root)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            prepared = root / "data/snli" / config["dataset"]["revision"] / f"cleaned-{CLEANING_VERSION}"
            example_splits().save_to_disk(str(prepared))
            threads = torch.get_num_threads()
            partial = run_cross_validation(config_path, max_fits=2)
            self.assertEqual(partial["manifest"]["status"], "incomplete")
            self.assertEqual(partial["manifest"]["completed_units"], 2)
            self.assertEqual(torch.get_num_threads(), threads)
            with self.assertRaisesRegex(ValueError, "2/33"):
                load_cv_report(config_path)
            completed = sorted(partial["run_dir"].glob("units/*/*/*/complete.json"))
            original_bytes = {p: p.read_bytes() for p in completed}
            with patch("part1.snli_experiments.fit_classifier", side_effect=RuntimeError("simulated interruption")):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run_cross_validation(config_path)
            self.assertEqual(torch.get_num_threads(), threads)
            failed_state = next(p for p in partial["run_dir"].glob("units/*/*/*/attempt_*/state.json")
                                if json.loads(p.read_text())["status"] == "failed")
            failed_bytes = failed_state.read_bytes()
            with patch("part1.snli_experiments.fit_classifier", wraps=fit_classifier) as fit:
                report = run_cross_validation(config_path)
            self.assertEqual(fit.call_count, 31)
            for call in fit.call_args_list:
                self.assertEqual(call.args[1].shape[0], 36)  # Training only, versus 18 held-out.
                self.assertEqual(len(call.args[2]), 36)
                self.assertNotIn("epoch_observer", call.kwargs)
            self.assertEqual(failed_state.read_bytes(), failed_bytes)
            self.assertTrue(all(p.read_bytes() == before for p, before in original_bytes.items()))
            self.assertEqual(report["manifest"]["status"], "complete")
            self.assertEqual(report["manifest"]["primary_fits"], 27)
            self.assertEqual(report["manifest"]["baseline_fits"], 6)
            self.assertTrue(all(report["manifest"]["checks"].values()))
            self.assertTrue(report["tables"]["summary"]["folds"].eq(3).all())
            self.assertEqual(report["tables"]["paired_accuracy"].shape, (3, 14))
            self.assertTrue(report["tables"]["fold_metrics"]["reload_max_abs_difference"].eq(0).all())
            self.assertEqual(torch.get_num_threads(), threads)
            protocol = prepare_evaluation_protocol(config_path)
            for seed in config["seeds"]["cv_splits"]:
                for system in planned_systems(config):
                    rows = report["tables"]["fold_metrics"].loc[
                        report["tables"]["fold_metrics"]["seed"].eq(seed)
                        & report["tables"]["fold_metrics"]["system_id"].eq(system["system_id"])
                    ]
                    predictions = pd.concat([pd.read_parquet(p) for p in rows["predictions"]])
                    self.assertEqual(len(predictions), 54)
                    self.assertEqual(predictions["example_id"].nunique(), 54)
                    self.assertTrue(predictions["source_split"].eq("train").all())
                    self.assertEqual(set(predictions["example_id"]), set(protocol["cv_pool"]["example_id"]))
            with self.assertRaisesRegex(ValueError, "complete paired"):
                summarize_metrics(report["tables"]["fold_metrics"].iloc[:-1], planned_systems(config), [0], 3)
            with patch("part1.snli_experiments.fit_classifier", side_effect=AssertionError("No retraining")), \
                 patch("part1.snli_experiments.prepare_fold_features", side_effect=AssertionError("No feature refitting")):
                reused = run_cross_validation(config_path)
                loaded = load_cv_report(config_path)
            self.assertEqual(reused["manifest_path"], report["manifest_path"])
            pd.testing.assert_frame_equal(loaded["tables"]["summary"], report["tables"]["summary"])
            predictions = Path(report["tables"]["fold_metrics"].iloc[0]["predictions"])
            predictions.write_bytes(predictions.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_cv_report(config_path)


if __name__ == "__main__":
    unittest.main()
