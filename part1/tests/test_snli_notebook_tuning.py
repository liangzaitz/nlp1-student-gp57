"""Exercise the preserved detailed notebook implementation and its historical runners."""
import ast
import json
import sys
import types
from pathlib import Path

import nbformat
import numpy as np
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
CONSTANTS = {
    "NOTEBOOK_PATH", "DEFAULT_CONFIG", "FROZEN_CONFIG", "ARTIFACT_FOLDERS",
    "SPLITS", "LABEL_NAMES", "TEXT_COLUMNS", "ID_COLUMNS", "CLEANING_VERSION",
    "PROTOCOL_VERSION", "POOL_FLAGS", "ROW_COLUMNS", "FEATURE_VERSION", "REPRESENTATIONS",
    "MODEL_VERSION", "CLASSIFIERS", "CLASS_IDS", "CV_VERSION", "PROBABILITY_COLUMNS",
    "PREDICTION_ROWS", "ANALYSIS_POLICY", "KEYS", "TABLES", "FAMILIES", "LABELS", "TUNING_CONFIG", "EXTENDED_TUNING_CONFIG",
}


def load_notebook_definitions():
    module = types.ModuleType("_snli_notebook_under_test")
    module.REPO_ROOT = ROOT
    sys.modules[module.__name__] = module
    saved = nbformat.read(ROOT / "part1/practical_1_details.ipynb", 4)
    for index, cell in enumerate(saved.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not node.level and not (node.module or "").startswith("part1")
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith("part1") for alias in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "compile", "__import__"}
        # Definition extraction is confined to this external verification harness.
        # Notebook users execute the literal cells normally.
        for node in tree.body:
            include = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
            if isinstance(node, ast.Assign):
                include = all(isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets)
            if include:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             f"<notebook-cell-{index}>", "exec"), module.__dict__)
    return module


def check_joint_schedule_covers_tokenizer_space_and_all_model_values(notebook):
    n = notebook
    base = json.loads(n.FROZEN_CONFIG.read_text())
    policy = json.loads(n.EXTENDED_TUNING_CONFIG.read_text())
    schedule = n.make_tuning_candidates(base, policy)
    assert schedule == n.make_tuning_candidates(base, policy)
    assert len(schedule) == 216
    for representation in n.REPRESENTATIONS:
        for kind in n.CLASSIFIERS:
            group = [c for c in schedule if c["representation"] == representation
                     and c["classifier"] == kind]
            assert len({tuple(c["tokenizer"][k] for k in
                              ("vocab_size", "min_frequency", "normalization")) for c in group}) == 24
            assert group[0]["baseline_hyperparameters"]
            if kind == "logistic":
                assert {c["model"]["C"] for c in group} == set(policy["space"]["C"])
            else:
                for key in ("learning_rate", "dropout", "weight_decay", "batch_size"):
                    assert {c["model"][key] for c in group} == set(policy["space"][key])
                # Independently shuffle balanced levels; repeating one permutation
                # would lock each learning rate to the same dropout in every cycle.
                assert len({(c["model"]["learning_rate"], c["model"]["dropout"])
                            for c in group}) >= 8


def check_sparse_features_and_serialized_normalization(notebook, tmp_path, normalization):
    n = notebook
    base = json.loads(n.FROZEN_CONFIG.read_text())
    encoder, _ = n.fit_pair_features(
        ["A CAT", "A cat", "Dog!"] * 3, ["cat CAT", "NO dog", "a cat."] * 3,
        {**base["tokenizer"], "vocab_size": 48, "normalization": normalization},
        {"min_frequency": 1, "max_features": 30})
    tokenized = n.tokenize_pairs(encoder.tokenizer, ["A CAT", "", "a Cat", ""],
                                ["", "CAT", "dog!", ""])
    reference = encoder.representations(encoder.count_matrix(tokenized))
    encoder.save(tmp_path)
    reloaded = n.PairFeatureEncoder.load(tmp_path)
    for representation in n.REPRESENTATIONS:
        actual = n.tuning_sparse_matrix(encoder, tokenized, representation)
        assert actual.shape == reference[representation].shape
        assert (actual != reference[representation]).nnz == 0
        assert (encoder.transform(["A CAT"], ["No cat!"], representation)
                != reloaded.transform(["A CAT"], ["No cat!"], representation)).nnz == 0
    if normalization == "lowercase":
        assert encoder.tokenizer.encode("A CAT").ids == encoder.tokenizer.encode("a cat").ids


def check_early_stopping_restores_best_weights(notebook, monkeypatch):
    n = notebook
    config = json.loads(n.FROZEN_CONFIG.read_text())
    config["neural_training"].update(num_threads=1, batch_size=3)
    config["neural_training"]["hidden_dims"] = {"mlp_one_hidden": [8], "mlp_two_hidden": [8, 4]}
    policy = json.loads(n.EXTENDED_TUNING_CONFIG.read_text())
    X = sparse.csr_matrix(np.tile(np.eye(3, dtype=np.float32), (4, 1)))
    y = np.tile(np.arange(3), 4)
    snapshots = []
    original_predict = n.FittedClassifier.predict_proba

    def capture(model, matrix, **kwargs):
        snapshots.append({key: value.detach().clone() for key, value in model.estimator.state_dict().items()})
        return original_predict(model, matrix, **kwargs)

    curve = iter([(.6, 1.), (.7, .9), (.65, .8), (.64, .7), (.63, .6)])

    def artificial_validation(labels, probabilities):
        accuracy, nll = next(curve)
        return {"accuracy": accuracy, "nll": nll, "macro_f1": accuracy}

    monkeypatch.setattr(n.FittedClassifier, "predict_proba", capture)
    monkeypatch.setattr(n, "tuning_validation_metrics", artificial_validation)
    fitted = n.fit_tuning_candidate("mlp_one_hidden", X, y, X, y, config, policy, seed=0)
    assert fitted.fit_info["best_epoch"] == 2
    assert fitted.fit_info["epochs_completed"] == 5
    for key, value in fitted.estimator.state_dict().items():
        assert n.torch.equal(value, snapshots[1][key])


def check_search_resume_full_finalists_and_corruption_guard(notebook, tmp_path, monkeypatch):
    n = notebook
    base = json.loads(n.FROZEN_CONFIG.read_text())
    base["paths"] = {"data_dir": str(tmp_path / "data"), "output_dir": str(tmp_path / "baseline")}
    base["tokenizer"]["vocab_size"] = 64
    base["neural_training"]["hidden_dims"] = {"mlp_one_hidden": [8], "mlp_two_hidden": [8, 4]}
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    monkeypatch.setattr(n, "FROZEN_CONFIG", base_path)
    policy = json.loads(n.EXTENDED_TUNING_CONFIG.read_text())
    policy.update(output_dir=str(tmp_path / "search"), screening_target_rows=18,
                  model_seeds=[0, 3], full_training_finalists_per_system=1,
                  initial_max_epochs=2, absolute_max_epochs=4, patience=1)
    policy["space"].update(vocab_size=[64], merge_min_frequency=[2],
                           one_hidden=[8], two_hidden=[[8, 4]], batch_size=[6])
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))

    def split(name, groups):
        labels = [label for _ in range(groups) for label in range(3)]
        raw = n.Dataset.from_dict({"premise": [f"{name} scene {i}" + (" ☃" if name == "validation" else "")
                                                          for i in range(groups) for _ in range(3)],
                                  "hypothesis": [f"{'yes maybe no'.split()[label]} animal" for label in labels],
                                  "label": labels},
                                 features=n.Features({"premise": n.Value("string"), "hypothesis": n.Value("string"),
                                                      "label": n.ClassLabel(names=list(n.LABEL_NAMES))}))
        return n.clean_split(raw, name)[0]

    cleaned = n.DatasetDict({"train": split("train", 12), "validation": split("validation", 3),
                             "test": split("test", 3)})
    prepared = tmp_path / "data/snli" / base["dataset"]["revision"] / f"cleaned-{n.CLEANING_VERSION}"
    prepared.parent.mkdir(parents=True)
    cleaned.save_to_disk(str(prepared))
    partial = n.run_tokenizer_model_search(policy_path, max_new_trials=2)
    assert partial["manifest"]["status"] == "incomplete"
    assert len(partial["records"]) == 2
    previous = {p: p.read_bytes() for p in partial["run_dir"].glob("*/trials/*/trial.json")}
    complete = n.run_tokenizer_model_search(policy_path)
    assert complete["manifest"]["status"] == "complete"
    assert len(complete["manifest"]["selected"]) == 9
    assert complete["manifest"]["eligible_training_rows"] == 36
    assert complete["manifest"]["screening_rows"] == 18
    assert complete["manifest"]["validation_rows"] == 9
    for path, content in previous.items():
        assert path.read_bytes() == content
    for record in complete["records"]:
        assert record["validation_rows"] == 9 and record["test_scored"] is False
        if record["stage"] == "full_training":
            assert record["training_rows"] == 36
    for path in complete["run_dir"].glob("*/tokenizers/*/encoder/tokenizer.json"):
        tokenizer = n.Tokenizer.from_file(str(path))
        assert tokenizer.token_to_id("☃") is None, "Validation-only text entered BPE fitting"
    for selection in complete["manifest"]["selected"]:
        expected = [0] if selection["classifier"] == "logistic" else [0, 3]
        assert selection["model_seeds"] == expected
    replay = n.load_tuning_report(policy_path)
    assert len(replay["records"]) == len(complete["records"])
    assert n.run_tokenizer_model_search(policy_path, max_new_trials=1)["manifest"]["status"] == "complete"
    # No held-out fold is needed by development selection; reject partial outer groups.
    frame = n.make_development_pools(cleaned)["cv_pool"].iloc[:1]
    with unittest.TestCase().assertRaisesRegex(ValueError, "complete premise groups"):
        n.run_tokenizer_model_search(policy_path, training_frame=frame, context="outer-seed0-fold0")
    predictions = next(complete["run_dir"].glob("*/trials/*/validation_predictions.npz"))
    with np.load(predictions, allow_pickle=False) as stored:
        assert all(value.startswith("validation:") for value in stored["example_id"])
    predictions.write_bytes(predictions.read_bytes() + b"changed")
    with unittest.TestCase().assertRaisesRegex(ValueError, "Changed or missing"):
        n.load_tuning_report(policy_path)


class NotebookTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = load_notebook_definitions()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp_path = Path(self.temporary.name)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.monkeypatch = types.SimpleNamespace(
            setattr=lambda obj, key, value: self.patches.enter_context(patch.object(obj, key, value)))

    def test_joint_schedule(self):
        check_joint_schedule_covers_tokenizer_space_and_all_model_values(self.notebook)

    def test_preserved_case_sparse_features(self):
        check_sparse_features_and_serialized_normalization(self.notebook, self.tmp_path, "none")

    def test_lowercase_sparse_features(self):
        check_sparse_features_and_serialized_normalization(self.notebook, self.tmp_path, "lowercase")

    def test_validation_checkpoint(self):
        check_early_stopping_restores_best_weights(self.notebook, self.monkeypatch)

    def test_selection_uses_mean_over_all_predeclared_seeds(self):
        n = self.notebook
        policy = json.loads(n.EXTENDED_TUNING_CONFIG.read_text())
        rows = []
        for candidate, scores in (("lucky", [.9, .5, .5]), ("stable", [.7, .7, .7])):
            for seed, accuracy in zip(policy["model_seeds"], scores):
                rows.append({"stage": "full_training", "representation": "indicators",
                             "classifier": "mlp_one_hidden", "candidate_id": candidate,
                             "seed": seed, "accuracy": accuracy, "nll": 1., "eligible": True,
                             "best_epoch": 3, "needs_ceiling_review": False})
        selected = n.tuning_selection(rows, policy)
        self.assertEqual(selected[0]["candidate_id"], "stable")
        self.assertAlmostEqual(selected[0]["mean_validation_accuracy"], .7)
        self.assertEqual(n.tuning_selection(rows[:2], policy), [])

    def test_complete_search_resume(self):
        check_search_resume_full_finalists_and_corruption_guard(
            self.notebook, self.tmp_path, self.monkeypatch)


    def test_wall_clock_limit_and_handler_restoration(self):
        import signal
        import time
        original = signal.getsignal(signal.SIGALRM)
        started = time.perf_counter()
        with self.assertRaises(self.notebook.TuningBudgetExpired):
            with self.notebook.tuning_time_limit(.03):
                time.sleep(.3)
        self.assertLess(time.perf_counter() - started, 1)
        self.assertEqual(signal.getsignal(signal.SIGALRM), original)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_budgeted_reuses_checked_trials_and_resumes(self):
        n = self.notebook
        base = json.loads(n.FROZEN_CONFIG.read_text())
        base["paths"] = {"data_dir": str(self.tmp_path / "data"),
                         "output_dir": str(self.tmp_path / "baseline")}
        base["tokenizer"]["vocab_size"] = 64
        base["neural_training"]["hidden_dims"] = {"mlp_one_hidden": [8], "mlp_two_hidden": [8, 4]}
        base_path = self.tmp_path / "base.json"
        base_path.write_text(json.dumps(base))
        self.monkeypatch.setattr(n, "FROZEN_CONFIG", base_path)
        old = json.loads(n.EXTENDED_TUNING_CONFIG.read_text())
        old.update(output_dir=str(self.tmp_path / "source"), screening_target_rows=18,
                   initial_max_epochs=2, absolute_max_epochs=4, patience=1)
        old["space"].update(vocab_size=[64], merge_min_frequency=[2],
                             one_hidden=[8], two_hidden=[[8, 4]], batch_size=[6])
        old_path = self.tmp_path / "extended.json"
        old_path.write_text(json.dumps(old))
        def split(name, groups):
            labels = [label for _ in range(groups) for label in range(3)]
            data = n.Dataset.from_dict({
                "premise": [f"{name} scene {i}" for i in range(groups) for _ in range(3)],
                "hypothesis": [f"{'yes maybe no'.split()[label]} animal" for label in labels],
                "label": labels}, features=n.Features({"premise": n.Value("string"),
                    "hypothesis": n.Value("string"), "label": n.ClassLabel(names=list(n.LABEL_NAMES))}))
            return n.clean_split(data, name)[0]
        data = n.DatasetDict({"train": split("train", 12), "validation": split("validation", 3),
                              "test": split("test", 3)})
        prepared = self.tmp_path / "data/snli" / base["dataset"]["revision"] / "cleaned-v1"
        prepared.parent.mkdir(parents=True)
        data.save_to_disk(str(prepared))
        source = n.run_tokenizer_model_search(old_path, max_new_trials=9)
        nbformat.write(nbformat.read(n.NOTEBOOK_PATH, 4), source["run_dir"] / "archived_search_execution.ipynb")
        originals = {p: p.read_bytes() for p in source["run_dir"].glob("coarse/trials/*/trial.json")}
        policy = json.loads(n.TUNING_CONFIG.read_text())
        policy.update(output_dir=str(self.tmp_path / "quick"), time_budget_seconds=30,
                      minimum_start_seconds=0, reuse_run_dir=str(source["run_dir"]),
                      screening_target_rows=18, initial_max_epochs=4, absolute_max_epochs=4, patience=1)
        policy["candidates"][1]["tokenizer"]["vocab_size"] = 48
        policy["candidates"][2]["tokenizer"]["vocab_size"] = 96
        for variant in policy["candidates"][1:]:
            variant["neural_training"].update(batch_size=6,
                hidden_dims={"mlp_one_hidden": [8], "mlp_two_hidden": [8, 4]})
        path = self.tmp_path / "quick.json"
        path.write_text(json.dumps(policy))
        partial = n.run_tokenizer_model_search(path, max_new_trials=1)
        self.assertEqual(partial["manifest"]["reused_trials"], 9)
        self.assertEqual(partial["manifest"]["completed_trials"], 10)
        self.assertEqual(partial["manifest"]["status"], "incomplete")
        complete = n.run_tokenizer_model_search(path)
        self.assertEqual(complete["manifest"]["status"], "complete")
        self.assertEqual(complete["manifest"]["completed_trials"], 27)
        self.assertFalse(complete["manifest"]["full_training_confirmation"])
        self.assertFalse(complete["manifest"]["multiple_seed_confirmation"])
        self.assertEqual(complete["manifest"]["screening_rows"], 18)
        self.assertEqual(complete["manifest"]["validation_rows"], 9)
        self.assertTrue(all(row["stage"] == "budgeted" and row["training_rows"] == 18
                            for row in complete["records"]))
        report = n.load_tuning_report(path)
        self.assertEqual(len(report["records"]), 27)
        self.assertTrue(all(row["evaluated_candidates"] == 3 for row in report["selected"]))
        self.assertTrue((complete["run_dir"] / "selected_recipes.json").exists())
        for original, contents in originals.items():
            self.assertEqual(original.read_bytes(), contents)
        # A timed-out active fit is excluded, while imported completed observations survive.
        policy["output_dir"] = str(self.tmp_path / "expired")
        path.write_text(json.dumps(policy))
        def expire(*args, **kwargs):
            raise n.TuningBudgetExpired("simulated expiry during active fit")
        self.monkeypatch.setattr(n, "run_tuning_trial", expire)
        expired = n.run_tokenizer_model_search(path)
        self.assertEqual(expired["manifest"]["status"], "budget_exhausted")
        self.assertEqual(expired["manifest"]["completed_trials"], 9)
        self.assertTrue(all(row["reused"] for row in expired["records"]))
        self.assertEqual(len(n.load_tuning_report(path)["records"]), 9)
        # Missing/incompatible implementation snapshots prevent cross-source reuse.
        (source["run_dir"] / "archived_search_execution.ipynb").unlink()
        policy["output_dir"] = str(self.tmp_path / "no_import")
        path.write_text(json.dumps(policy))
        no_import = n.run_tokenizer_model_search(path)
        self.assertEqual(no_import["manifest"]["reused_trials"], 0)
        self.assertEqual(no_import["manifest"]["status"], "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
