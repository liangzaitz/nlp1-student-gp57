"""Pairing, corrected posterior mathematics, planned contrasts and saved reports."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import baycomp
import numpy as np
import pandas as pd
from scipy.stats import t

from part1.snli_analysis import (
    correlated_posterior, load_analysis_report,
    paired_scores, planned_contrasts, run_bayesian_analysis,
)
from part1.snli_experiments import planned_systems
from part1.snli_pilot import FROZEN_CONFIG


def config_and_metrics():
    config = json.loads(FROZEN_CONFIG.read_text())
    rows = []
    for repetition, seed in enumerate(config["seeds"]["cv_splits"]):
        for fold in range(5):
            for i, system in enumerate(planned_systems(config)):
                rows.append(system | {"repetition": repetition, "seed": seed, "fold": fold,
                                      "accuracy": .4 + .01 * i + .001 * fold + .0005 * repetition * i,
                                      "training_rows": 80, "heldout_rows": 20,
                                      "feature_id": f"{seed}_{fold}", "run_id": "example_run",
                                      "protocol_id": "example_protocol", "config_sha256": "example_config", "model_seed": 0})
    return config, pd.DataFrame(rows)


class PairingTests(unittest.TestCase):
    def test_alignment_uses_keys_and_rejects_malformed_or_unpaired_scores(self):
        config, metrics = config_and_metrics()
        expected = paired_scores(metrics, config)
        pd.testing.assert_frame_equal(expected, paired_scores(metrics.sample(frac=1, random_state=42), config))
        self.assertEqual(expected.shape, (15, 14))
        for change in ("missing", "duplicate", "wrong_seed", "feature", "sizes", "percent", "nan", "metadata", "pool", "provenance"):
            bad = metrics.copy()
            if change == "missing":
                bad = bad.iloc[:-1]
            elif change == "duplicate":
                bad = pd.concat([bad, bad.iloc[:1]])
            elif change == "wrong_seed":
                bad.loc[0, "seed"] = 999
            elif change == "feature":
                bad.loc[0, "feature_id"] = "another_tokenizer"
            elif change == "sizes":
                bad.loc[0, "heldout_rows"] = 10
            elif change == "percent":
                bad.loc[0, "accuracy"] = 50
            elif change == "nan":
                bad.loc[0, "accuracy"] = np.nan
            elif change == "metadata":
                bad.loc[0, "input_mode"] = "hypothesis_only"
            elif change == "pool":
                bad["training_rows"] = 81
            else:
                bad.loc[0, "protocol_id"] = "other_protocol"
            with self.assertRaises(ValueError, msg=change):
                paired_scores(bad, config)

    def test_contrasts_are_complete_one_factor_pairs_and_fixed_context(self):
        config, _ = config_and_metrics()
        contrasts = planned_contrasts(config)
        self.assertEqual(len(contrasts), 28)
        self.assertEqual(int(contrasts["primary"].sum()), 18)
        self.assertEqual(contrasts["contrast_id"].nunique(), 28)
        for row in contrasts.to_dict("records"):
            a, b = row["system_a"].split("__"), row["system_b"].split("__")
            if row["family"] == "representation":
                self.assertNotEqual(a[0], b[0])
                self.assertEqual(a[1:], b[1:])
            elif row["family"] == "architecture":
                self.assertEqual(a[0], b[0])
                self.assertNotEqual(a[1], b[1])
                self.assertEqual(a[2], b[2])
        ablation = contrasts.iloc[-1]
        self.assertEqual(ablation["system_a"], "unigram_counts__logistic__pair")
        self.assertEqual(ablation["system_b"], "unigram_counts__logistic__hypothesis_only")


class PosteriorTests(unittest.TestCase):
    def setUp(self):
        self.a = np.linspace(.55, .65, 15)
        self.b = self.a + np.tile([-.002, .003, .006, .014, .009], 3)

    def test_corrected_scale_intervals_probabilities_and_actual_standard_deviation(self):
        result = correlated_posterior(self.a, self.b, runs=3, rope=.005)
        variance = np.var(self.b - self.a, ddof=1)
        scale = np.sqrt(variance * (1 / 15 + 1 / 4))
        posterior = baycomp.CorrelatedTTest(self.a, self.b, rope=.005, runs=3)
        self.assertAlmostEqual(result["student_scale"], scale)
        self.assertAlmostEqual(result["posterior_std"], scale * np.sqrt(14 / 12))
        self.assertAlmostEqual(result["student_scale"] ** 2 / (variance / 15), 4.75)
        self.assertEqual((result["runs"], result["folds_per_run"], result["df"]), (3, 5, 14))
        np.testing.assert_allclose([result["p_a_better"], result["p_equivalent"], result["p_b_better"]], posterior.probs())
        np.testing.assert_allclose([result["ci_low"], result["ci_high"]], t.interval(.95, 14, loc=np.mean(self.b - self.a), scale=scale))
        self.assertNotAlmostEqual(result["student_scale"], np.sqrt(variance * (1 / 15 + 1 / 14)))  # Wrong runs=1.

    def test_swapping_systems_reverses_direction_and_rope_sensitivity_preserves_posterior(self):
        forward = correlated_posterior(self.a, self.b, runs=3, rope=.005)
        reverse = correlated_posterior(self.b, self.a, runs=3, rope=.005)
        self.assertAlmostEqual(forward["mean_difference"], -reverse["mean_difference"])
        self.assertAlmostEqual(forward["ci_low"], -reverse["ci_high"])
        self.assertAlmostEqual(forward["p_a_better"], reverse["p_b_better"])
        self.assertAlmostEqual(forward["p_equivalent"], reverse["p_equivalent"])
        wider = correlated_posterior(self.a, self.b, runs=3, rope=.01)
        self.assertGreater(wider["p_equivalent"], forward["p_equivalent"])
        self.assertEqual(wider["mean_difference"], forward["mean_difference"])
        self.assertEqual(wider["student_scale"], forward["student_scale"])

    def test_point_masses_including_rope_boundary_have_finite_probabilities(self):
        a = np.full(15, .5)
        for difference, rope, expected in ((0, .005, (0, 1, 0)), (.03125, .005, (0, 0, 1)),
                                           (-.03125, .005, (1, 0, 0)), (.03125, .03125, (0, 1, 0))):
            result = correlated_posterior(a, a + difference, runs=3, rope=rope)
            self.assertTrue(result["point_mass"])
            self.assertEqual(tuple(result[k] for k in ("p_a_better", "p_equivalent", "p_b_better")), expected)
            self.assertEqual(result["ci_low"], result["ci_high"])
            self.assertEqual(result["posterior_std"], 0)

    def test_invalid_inputs_are_rejected(self):
        for options in ({"runs": 4}, {"runs": 0}, {"runs": True}, {"rope": 0}, {"rope": np.nan}, {"credible_mass": 1}):
            kwargs = {"runs": 3, "rope": .005} | options
            with self.assertRaises(ValueError):
                correlated_posterior(self.a, self.b, **kwargs)
        with self.assertRaises(ValueError):
            correlated_posterior(self.a[:3], self.b[:3], runs=1, rope=.005)
        with self.assertRaises(ValueError):
            correlated_posterior(self.a * 100, self.b, runs=3, rope=.005)


class ReportTests(unittest.TestCase):
    def test_analysis_is_reusable_checked_and_does_not_train(self):
        from part1.snli_analysis import _sha
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, metrics = config_and_metrics()
            config["paths"]["output_dir"] = str(root / "results")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            metrics["config_sha256"] = _sha(config_path)
            source = root / "cv"
            source.mkdir()
            marker = source / "manifest.json"
            manifest = {"run_id": "example_run", "status": "complete", "specification": {"protocol_id": "example_protocol"}}
            marker.write_text(json.dumps(manifest))
            paired = paired_scores(metrics, config)
            for name, table in (("fold_metrics", metrics), ("paired_accuracy", paired)):
                table.to_csv(source / f"{name}.csv", index=False)
            cv = {"manifest": manifest, "manifest_path": marker, "run_dir": source,
                  "tables": {"fold_metrics": metrics, "paired_accuracy": paired}}
            with patch("part1.snli_analysis.load_cv_report", return_value=cv), \
                 patch("part1.snli_experiments.fit_classifier", side_effect=AssertionError("No fitting")), \
                 patch("part1.snli_analysis._plot_posterior", side_effect=lambda row, path: path.write_bytes(b"figure")), \
                 patch("part1.snli_analysis._plot_overview", side_effect=lambda rows, path: path.write_bytes(b"figure")):
                with self.assertRaises(FileNotFoundError):
                    load_analysis_report(config_path)
                report = run_bayesian_analysis(config_path)
                before = {p: p.read_bytes() for p in report["analysis_dir"].rglob("*") if p.is_file()}
                reused = run_bayesian_analysis(config_path)
                loaded = load_analysis_report(config_path)
                self.assertEqual(reused["manifest_path"], loaded["manifest_path"])
                self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))
                self.assertEqual(len(loaded["tables"]["posteriors"]), 28)
                self.assertEqual(len(loaded["tables"]["paired_differences"]), 28 * 15)
                self.assertEqual(len(loaded["tables"]["rope_sensitivity"]), 84)
                marker.write_text(json.dumps(manifest | {"changed_source": True}))
                with self.assertRaisesRegex(ValueError, "source/code/policy"):
                    load_analysis_report(config_path)
                marker.write_text(json.dumps(manifest))
                artifact = loaded["analysis_dir"] / "posteriors.csv"
                artifact.write_text(artifact.read_text() + "corrupted")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    run_bayesian_analysis(config_path)
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    load_analysis_report(config_path)
                cv["tables"]["paired_accuracy"] = paired.copy()
                cv["tables"]["paired_accuracy"].iloc[0, 3] += .1
                with self.assertRaisesRegex(ValueError, "paired accuracy"):
                    run_bayesian_analysis(config_path)
                for malformed in (pd.concat([paired, paired.iloc[:1]]), paired.iloc[:-1]):
                    cv["tables"]["paired_accuracy"] = malformed
                    with self.assertRaisesRegex(ValueError, "partition keys"):
                        run_bayesian_analysis(config_path)


if __name__ == "__main__":
    unittest.main()
