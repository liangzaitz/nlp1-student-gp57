"""Exercise the literal streamlined notebook definitions without running SNLI fits."""
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "part1/practical_1.ipynb"
CONSTANTS = {"PART1", "OUTPUT", "BASE", "SEARCH", "LABELS", "REPRESENTATIONS",
             "CLASSIFIERS", "PROBABILITY_COLUMNS", "KEYS", "NEW_RUNS", "MAX_SECONDS"}


def definitions():
    module = types.ModuleType("_streamlined_notebook_test")
    module.ROOT = ROOT
    sys.modules[module.__name__] = module
    for index, cell in enumerate(nbformat.read(NOTEBOOK, 4).cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not node.level and (node.module or "").split(".")[0] not in {
                    "part1", "snli_data", "snli_features", "snli_models", "snli_experiments",
                    "snli_evaluation", "snli_analysis", "snli_pilot", "project_setup", "smlp"}
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith(("part1", "snli_")) for a in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "compile", "__import__"}
        # Extraction belongs only to the external test harness. Student cells are literal Python.
        for node in tree.body:
            include = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
            if isinstance(node, ast.Assign):
                include = all(isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets)
            if include:
                exec(compile(ast.Module(body=[node], type_ignores=[]), f"<cell-{index}>", "exec"), module.__dict__)
    return module


class TestStreamlinedNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = definitions()
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def small_config(self):
        config = deepcopy(self.n.BASE)
        config["tokenizer"].update(vocab_size=64, min_frequency=1, normalization="lowercase")
        config["bigram_features"].update(max_features=32, min_frequency=1)
        config["neural_training"].update(batch_size=6, num_epochs=2,
            hidden_dims={"mlp_one_hidden": [8], "mlp_two_hidden": [8, 4]})
        return config

    def frame(self, prefix="train", groups=6):
        return pd.DataFrame({"premise": [f"{prefix} group {i} CAT" for i in range(groups) for _ in range(3)],
            "hypothesis": [f"{word} cat" for _ in range(groups) for word in ("yes", "maybe", "no")],
            "label": np.tile(np.arange(3), groups),
            "example_id": [f"{prefix}:{i}" for i in range(groups * 3)]})

    def test_features_match_direct_counts_including_empty_roles(self):
        n = self.n
        encoder, _ = n.PairEncoder.fit(self.frame(), self.small_config())
        frame = pd.DataFrame({"premise": ["CAT cat", "", "cat", "", ""],
                              "hypothesis": ["", "YES cat", "cat cat", "", "cat"]})
        pair_index = {pair: i for i, pair in enumerate(encoder.bigrams)}
        for representation in n.REPRESENTATIONS:
            width = len(pair_index) if representation == "bigram_counts" else encoder.vocab_size
            expected = np.zeros((len(frame), width * 2), np.float32)
            for row, texts in enumerate(frame.itertuples(index=False, name=None)):
                for role, text in enumerate(texts):
                    ids = encoder.tokenizer.encode(text).ids
                    counts = Counter(zip(ids, ids[1:])) if representation == "bigram_counts" else Counter(ids)
                    for key, value in counts.items():
                        if representation == "bigram_counts" and key not in pair_index:
                            continue
                        column = pair_index[key] if representation == "bigram_counts" else key
                        expected[row, role * width + column] = 1 if representation == "indicators" else value
            actual = encoder.transform(frame, representation)
            np.testing.assert_array_equal(actual.toarray(), expected)
        with tempfile.TemporaryDirectory() as directory:
            encoder.save(directory)
            loaded = n.PairEncoder.load(directory)
            self.assertEqual(loaded.tokenizer.encode("CAT").ids, loaded.tokenizer.encode("cat").ids)
            for representation in n.REPRESENTATIONS:
                self.assertEqual((loaded.transform(frame, representation) != encoder.transform(frame, representation)).nnz, 0)

    def test_fixed_training_matches_historical_implementation(self):
        from part1.snli_models import fit_classifier
        n, config = self.n, self.small_config()
        X = sparse.csr_matrix(np.tile(np.eye(3, dtype=np.float32), (6, 1)))
        y = np.tile(np.arange(3), 6)
        for kind in (*n.CLASSIFIERS, "class_prior"):
            with self.subTest(kind=kind):
                actual = n.fit_model(X, y, kind, config)
                expected = fit_classifier(kind, X, y, config, seed=0)
                np.testing.assert_allclose(n.predict_proba(actual, X), expected.predict_proba(X), atol=1e-7, rtol=0)
                self.assertEqual(n.predict_proba(actual, X[:0]).shape, (0, 3))
                if kind.startswith("mlp"):
                    for key, value in actual["estimator"].state_dict().items():
                        self.assertTrue(torch.equal(value, expected.estimator.state_dict()[key]), key)
        with self.assertRaises(AssertionError):
            n.fit_model(X, y.astype(float) + .1, "logistic", config)

    def test_early_stopping_restores_best_checkpoint(self):
        n = self.n
        X = sparse.csr_matrix(np.tile(np.eye(3, dtype=np.float32), (4, 1)))
        y = np.tile(np.arange(3), 4)
        snapshots, original = [], n.predict_proba
        def capture(model, matrix):
            snapshots.append({k: v.clone() for k, v in model["estimator"].state_dict().items()})
            return original(model, matrix)
        curve = iter([(.6, 1.), (.7, .9), (.65, .8), (.64, .7), (.63, .6)])
        def metric(*args):
            accuracy, nll = next(curve)
            return {"accuracy": accuracy, "nll": nll, "macro_f1": accuracy}
        with patch.object(n, "predict_proba", capture), patch.object(n, "score", metric):
            fitted = n.fit_model(X, y, "mlp_one_hidden", self.small_config(),
                                 validation=(X, y), epochs=8, patience=3)
        self.assertEqual(fitted["fit_info"]["best_epoch"], 2)
        self.assertEqual(fitted["fit_info"]["epochs_completed"], 5)
        for key, value in fitted["estimator"].state_dict().items():
            self.assertTrue(torch.equal(value, snapshots[1][key]))

    def test_raw_text_bundle_roundtrips_all_models_and_hypothesis_mode(self):
        n, frame, config = self.n, self.frame(), self.small_config()
        encoder, tokens = n.PairEncoder.fit(frame, config)
        X = n.feature_matrix(encoder, tokens, "unigram_counts")
        with tempfile.TemporaryDirectory() as directory:
            for kind, mode in [(k, "pair") for k in (*n.CLASSIFIERS, "class_prior")] + [("logistic", "hypothesis_only")]:
                selected = X if mode == "pair" else X[:, X.shape[1] // 2:]
                fitted = n.fit_model(selected, frame.label, kind, config)
                path = Path(directory) / f"{kind}_{mode}"
                n.save_pipeline(path, encoder, fitted, "unigram_counts", mode)
                loaded_encoder, loaded = n.load_pipeline(path)
                np.testing.assert_allclose(n.predict_text(loaded_encoder, loaded, frame), n.predict_proba(fitted, selected), atol=1e-12, rtol=0)
            weights = path / "model.joblib"
            weights.write_bytes(weights.read_bytes() + b"changed")
            with self.assertRaises(AssertionError):
                n.load_pipeline(path)

    def test_short_search_resumes_and_rejects_changed_labels(self):
        n, config = self.n, self.small_config()
        policy = {"candidates": [{"name": "small"}], "time_budget_seconds": 30,
                  "absolute_max_epochs": 2, "patience": 1}
        training, validation = self.frame(), self.frame("validation", 2)
        with tempfile.TemporaryDirectory() as directory:
            first = n.run_search(training, validation, directory, base=config, policy=policy)
            self.assertEqual(len(first), 9)
            before = {p: p.read_bytes() for p in Path(directory).rglob("result.json")}
            with patch.object(n, "fit_model", side_effect=AssertionError("Resume must reuse completed trials")):
                resumed = n.run_search(training, validation, directory, base=config, policy=policy)
            self.assertEqual(len(resumed), 9)
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            validation.loc[0, "label"] = (validation.loc[0, "label"] + 1) % 3
            with self.assertRaises(AssertionError):
                n.run_search(training, validation, directory, base=config, policy=policy)

    def test_small_grouped_cv_covers_11_systems_without_outer_selection(self):
        n, frame, config = self.n, self.frame(groups=8), self.small_config()
        metadata = frame[["example_id", "label"]].copy()
        metadata["clean_index"] = np.arange(len(frame))
        metadata["premise_group"] = frame.premise
        folds = n.grouped_folds(metadata, n_splits=2, seeds=(0,))
        data = DatasetDict({"train": Dataset.from_pandas(frame, preserve_index=False)})
        original = n.fit_model
        def train_only(*args, **kwargs):
            self.assertIsNone(kwargs.get("validation"))
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(n, "fit_model", train_only):
            path = Path(directory) / "cv"
            scores = n.run_cv(data, folds, path, config=config, seeds=(0,), n_splits=2, max_seconds=30)
            self.assertEqual(len(scores), 22)
            self.assertTrue(json.loads((path / "status.json").read_text())["complete"])
            self.assertTrue(scores.groupby("system_id").size().eq(2).all())
            for file in path.glob("*.parquet"):
                predictions = pd.read_parquet(file)
                self.assertEqual(len(predictions), len(frame) // 2)

    def test_correlated_posterior_matches_baycomp_and_degenerate_rope(self):
        from baycomp import CorrelatedTTest
        n = self.n
        a = np.array([.6, .61, .59, .62, .60] * 3)
        b = a + np.array([.01, -.01, .005, .02, -.005] * 3)
        expected = CorrelatedTTest(a, b, rope=.005, runs=3)
        actual = n.correlated_posterior(a, b)
        np.testing.assert_allclose([actual[k] for k in ("p_a_better", "p_equivalent", "p_b_better")], expected.probs(), atol=1e-12, rtol=0)
        self.assertEqual(n.correlated_posterior(a, a)["p_equivalent"], 1)
        self.assertEqual(n.correlated_posterior(a, a + .1)["p_b_better"], 1)

    def test_real_budget_interrupts_and_restores_timer(self):
        n = self.n
        old_handler = n.signal.getsignal(n.signal.SIGALRM)
        with self.assertRaises(n.BudgetExpired):
            with n.time_budget(.02):
                time.sleep(.2)
        self.assertEqual(n.signal.getsignal(n.signal.SIGALRM), old_handler)
        self.assertEqual(n.signal.getitimer(n.signal.ITIMER_REAL)[0], 0)
        with self.assertRaises(ValueError):
            with n.time_budget(901):
                pass

    def test_oof_alignment_rejects_duplicates_missing_rows_and_wrong_gold(self):
        n = self.n
        assignments = pd.DataFrame({"example_id": [f"train:{i}" for i in range(4)],
            "clean_index": range(4), "label": [0, 1, 2, 0],
            "premise_group": [f"group{i}" for i in range(4)], "fold_seed_0": [0, 0, 1, 1]})
        system = "example_system"
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for fold in range(2):
                saved = assignments.loc[assignments.fold_seed_0.eq(fold)].copy()
                saved["fold"], saved["seed"] = fold, 0
                saved["prediction"] = saved.label
                saved[n.PROBABILITY_COLUMNS] = np.eye(3)[saved.label] * .8 + .2 / 3
                path = Path(directory) / f"fold{fold}.parquet"
                saved.iloc[::-1].to_parquet(path, index=False)
                paths.append(path)
            metrics = pd.DataFrame({"system_id": [system] * 2, "seed": [0, 0], "predictions": paths})
            rows, probabilities, scores = n.collect_oof(metrics, assignments)
            self.assertEqual(rows.example_id.tolist(), assignments.example_id.tolist())
            self.assertEqual(scores.accuracy.iloc[0], 1)
            self.assertEqual(probabilities[system].shape, (4, 3))
            with self.assertRaises(AssertionError):
                n.collect_oof(metrics.iloc[:1], assignments)
            original = pd.read_parquet(paths[0])
            duplicate = original.copy()
            duplicate.loc[0, "example_id"] = duplicate.loc[1, "example_id"]
            duplicate.to_parquet(paths[0], index=False)
            with self.assertRaises(AssertionError):
                n.collect_oof(metrics, assignments)
            wrong_gold = original.copy()
            wrong_gold.loc[0, "label"] = 2
            wrong_gold.to_parquet(paths[0], index=False)
            with self.assertRaises(AssertionError):
                n.collect_oof(metrics, assignments)

    def test_error_slices_and_shared_counts_use_explicit_denominators(self):
        n = self.n
        rows = pd.DataFrame({"premise": ["a dog runs", "a dog does not run", "notable cats"],
            "hypothesis": ["a dog runs", "a dog runs", "cats"], "label": [0, 2, 1],
            "anchor": [0, 0, 2], "comparator": [0, 2, 0]})
        rows = pd.concat([rows, n.text_diagnostics(rows)], axis=1)
        self.assertEqual(rows.negation_cue.tolist(), [False, True, False])
        np.testing.assert_allclose(rows.hypothesis_overlap, [1, 2/3, 1])
        slices = n.slice_scores(rows, {"Anchor": "anchor"}).set_index("slice")
        self.assertEqual(slices.loc["All rows", "rows"], 3)
        self.assertAlmostEqual(slices.loc["All rows", "Anchor"], 1/3)
        self.assertEqual(slices.loc["High overlap (≥.8)", "rows"], 2)
        self.assertEqual(slices.loc["High overlap, non-entailment", "Anchor"], 0)
        shared = n.shared_errors(rows, "anchor", {"Other": "comparator"}).iloc[0]
        self.assertEqual([shared[k] for k in ("both_correct", "both_wrong", "anchor_wrong_only", "comparator_wrong_only")], [1, 1, 1, 0])
        self.assertEqual(shared.same_wrong_label, 0)

    def test_error_review_sampling_is_deterministic_and_premise_disjoint(self):
        n, records = self.n, []
        for gold in range(3):
            for predicted in range(3):
                if gold != predicted:
                    records.extend({"label": gold, "anchor": predicted, "bigram": gold,
                                    "hypothesis": gold, "nn2": predicted, "ind2": predicted} for _ in range(20))
        records.extend({"label": i % 3, "anchor": i % 3, "bigram": (i+1) % 3,
                        "hypothesis": i % 3, "nn2": i % 3, "ind2": i % 3} for i in range(10))
        rows = pd.DataFrame(records)
        rows["example_id"] = [f"train:{i}" for i in range(len(rows))]
        rows["premise_group"] = [f"group{i//2}" for i in range(len(rows))]
        rows["anchor_confidence"] = np.linspace(.4, .99, len(rows))
        arguments = (rows, "anchor", "bigram", "hypothesis", ["nn2", "ind2"])
        review = n.choose_error_cases(*arguments)
        pd.testing.assert_frame_equal(review, n.choose_error_cases(*arguments))
        self.assertEqual(len(review), 50)
        self.assertTrue(review.example_id.is_unique and review.premise_group.is_unique)
        self.assertTrue(review.selection_reason.value_counts().eq(5).all())
        rescued = review.loc[review.selection_reason.eq("Bigram correct, anchor wrong")]
        self.assertTrue(rescued.bigram.eq(rescued.label).all() and rescued.anchor.ne(rescued.label).all())

    def final_data(self):
        frames = {name: self.frame(name, groups) for name, groups in
                  [("train", 4), ("validation", 2), ("test", 2)]}
        frames["validation"].loc[:2, "premise"] = frames["train"].premise.iloc[0]
        frames["test"].loc[:2, "premise"] = frames["train"].premise.iloc[3]
        frames["test"]["hypothesis"] += " ☃"
        for name, frame in frames.items():
            frame["source_split"], frame["source_row"] = name, np.arange(len(frame))
        data = DatasetDict({name: Dataset.from_pandas(frame, preserve_index=False) for name, frame in frames.items()})
        config = self.small_config()
        config.update(representation="unigram_counts", classifier="mlp_one_hidden")
        config["final_training"] = {"expected_rows": 15, "max_seconds": 30}
        return data, config

    def test_final_refit_recovers_validation_overlaps_and_scores_test_only_once(self):
        n = self.n
        data, config = self.final_data()
        training = n.final_training_frame(data)
        self.assertEqual(len(training), 15)
        self.assertTrue(training.premise.eq("train group 0 CAT").sum() == 6)
        self.assertFalse(training.premise.eq("train group 1 CAT").any())
        original, test_calls = n.predict_text, []
        def capture(encoder, model, frame):
            if frame.source_split.eq("test").all():
                test_calls.append(frame.example_id.tolist())
            return original(encoder, model, frame)
        with tempfile.TemporaryDirectory() as directory, patch.object(n, "predict_text", capture):
            manifest, saved = n.run_final_experiment(data, config, directory)
            self.assertEqual(len(test_calls), 1)
            self.assertEqual(manifest["training_rows"], 15)
            self.assertEqual(manifest["epochs"], 2)
            self.assertEqual(manifest["test_rows"], 6)
            self.assertFalse(manifest["test_used_for_selection"])
            encoder, model = n.load_pipeline(Path(directory) / "bundle")
            self.assertIsNone(encoder.tokenizer.token_to_id("☃"))
            with patch.object(n, "fit_model", side_effect=AssertionError("must not refit")), \
                 patch.object(n, "predict_text", side_effect=AssertionError("must not score test again")):
                replay, _ = n.run_final_experiment(data, config, directory)
            self.assertEqual(replay["metrics"], manifest["metrics"])
            path = Path(directory) / "test_predictions.parquet"
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaises(AssertionError):
                n.load_final_result(directory, config)

    def test_final_budget_failure_never_enters_test_and_test_journal_blocks_retries(self):
        n = self.n
        data, config = self.final_data()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(n, "fit_model", side_effect=n.BudgetExpired("simulated budget")):
            self.assertIsNone(n.run_final_experiment(data, config, directory))
            self.assertFalse((Path(directory) / "test_started.json").exists())
            self.assertFalse((Path(directory) / "manifest.json").exists())
        original = n.predict_text
        def fail_test(encoder, model, frame):
            if frame.source_split.eq("test").all():
                raise RuntimeError("simulated interruption at test inference")
            return original(encoder, model, frame)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(n, "predict_text", fail_test), self.assertRaises(RuntimeError):
                n.run_final_experiment(data, config, directory)
            with patch.object(n, "fit_model", side_effect=AssertionError("must not refit")), \
                 self.assertRaisesRegex(AssertionError, "already started"):
                n.run_final_experiment(data, config, directory)

    def test_confidence_bins_include_probability_one_and_preserve_argmax_ties(self):
        probabilities = np.array([[1., 0, 0], [.2, .4, .4], [.3, .5, .2]])
        result = self.n.confidence_bins([0, 2, 1], probabilities)
        self.assertEqual(result["count"].sum(), 3)
        self.assertEqual(result.iloc[-1].bin_right, 1)
        self.assertEqual(result.iloc[0].accuracy, 0)
        ece = np.average(abs(result.accuracy - result.confidence), weights=result["count"])
        self.assertAlmostEqual(ece, .3)


if __name__ == "__main__":
    unittest.main()
