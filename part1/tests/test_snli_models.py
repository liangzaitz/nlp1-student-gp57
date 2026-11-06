"""Numerical, training-boundary, baseline, and complete prediction-bundle checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.special import softmax
import torch
from torch import nn

from part1.project_setup import DEFAULT_CONFIG
from part1.snli_features import REPRESENTATIONS, fit_pair_features
from part1.snli_models import (
    CLASSIFIERS, PairMLP, build_neural_model, fit_classifier, load_prediction_bundle,
    save_prediction_bundle, scipy_batch_to_torch, score_probabilities,
    select_input_features, validate_probabilities,
)


def settings():
    config = deepcopy(json.loads(DEFAULT_CONFIG.read_text()))
    config["neural_training"]["hidden_dims"] = {"mlp_one_hidden": [8], "mlp_two_hidden": [8, 4]}
    config["neural_training"]["batch_size"] = 12
    config["device"] = "cpu"
    return config


class MetricTests(unittest.TestCase):
    def test_nll_matches_log_softmax_loss_and_decision_metrics(self):
        logits = torch.tensor([[3., 1., 0.], [0., 3., 1.], [1., 0., 3.]], dtype=torch.float64)
        targets = torch.arange(3)
        log_probabilities = torch.log_softmax(logits, dim=1)
        scores = score_probabilities(targets.numpy(), log_probabilities.exp().numpy())
        self.assertAlmostEqual(scores["nll"], nn.NLLLoss()(log_probabilities, targets).item())
        self.assertEqual(scores["accuracy"], 1.0)
        self.assertEqual(scores["macro_f1"], 1.0)
        self.assertEqual(scores["confusion_matrix"], np.eye(3, dtype=int).tolist())

    def test_uniform_ties_clipping_missing_predictions_and_invalid_values(self):
        scores = score_probabilities([0, 1, 2], np.full((3, 3), 1 / 3))
        self.assertAlmostEqual(scores["accuracy"], 1 / 3)
        self.assertAlmostEqual(scores["nll"], np.log(3))
        self.assertEqual(scores["confusion_matrix"], [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(scores["per_class"]["neutral"]["precision"], 0.0)
        zero = score_probabilities([1], [[1., 0., 0.]])
        self.assertAlmostEqual(zero["nll"], -np.log(np.finfo(np.float64).eps))
        for probabilities in ([[.5, .5]], [[.1, .1, .1]], [[-1., 1., 1.]], [[np.nan, 0., 1.]]):
            with self.assertRaises(ValueError):
                validate_probabilities(probabilities)
        with self.assertRaises(ValueError):
            score_probabilities([3], [[.3, .3, .4]])


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.config = settings()
        self.X = sparse.csr_matrix(np.tile(np.eye(3, dtype=np.float32), (4, 1)))
        self.y = np.tile(np.arange(3), 4)

    def test_sparse_forward_matches_dense_first_layer_and_has_gradients(self):
        for hidden in ([8], [8, 4]):
            model = PairMLP(3, hidden, dropout=0.4)
            model.eval()
            expected = model.output(torch.relu(torch.from_numpy(self.X.toarray()) @ model.weight + model.bias))
            with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("Dense input allocation")):
                actual = model(scipy_batch_to_torch(self.X))
            torch.testing.assert_close(actual, expected)
            loss = nn.NLLLoss()(actual, torch.from_numpy(self.y))
            loss.backward()
            self.assertIsNotNone(model.weight.grad)
            self.assertTrue(torch.isfinite(model.weight.grad).all())

    def test_logistic_softmax_parameters_and_convergence_status(self):
        model = fit_classifier("logistic", self.X, self.y, self.config)
        expected = softmax(self.X @ model.estimator.coef_.T + model.estimator.intercept_, axis=1)
        np.testing.assert_allclose(model.predict_proba(self.X), expected, atol=1e-6)
        self.assertEqual(model.parameter_count, 3 * 3 + 3)
        self.assertTrue(model.fit_info["converged"])
        self.config["logistic_training"]["max_iter"] = 1
        unfinished = fit_classifier("logistic", self.X, self.y, self.config)
        self.assertFalse(unfinished.fit_info["converged"])
        self.assertTrue(unfinished.fit_info["convergence_warnings"])

    def test_neural_training_reduces_loss_is_seeded_and_disables_dropout(self):
        self.config["neural_training"]["dropout"] = 0.0
        self.config["neural_training"]["learning_rate"] = 0.05
        for kind in CLASSIFIERS[1:]:
            with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("Dense input allocation")):
                model = fit_classifier(kind, self.X, self.y, self.config, num_epochs=20)
            self.assertLess(model.history[-1]["train_nll_online"], model.history[0]["train_nll_online"])
            self.assertEqual(model.fit_info["checkpoint_policy"], "last_fixed_epoch")
            self.assertEqual(model.fit_info["epochs_completed"], 20)
            first = model.predict_proba(self.X, batch_size=5)
            second = model.predict_proba(self.X, batch_size=5)
            np.testing.assert_array_equal(first, second)
            self.assertFalse(model.estimator.training)
            duplicate = fit_classifier(kind, self.X, self.y, self.config, num_epochs=20)
            np.testing.assert_array_equal(first, duplicate.predict_proba(self.X, batch_size=5))
            self.assertEqual(model.parameter_count, sum(parameter.numel() for parameter in model.estimator.parameters()))

    def test_eval_dropout_and_empty_predictions(self):
        model = fit_classifier("mlp_two_hidden", self.X, self.y, self.config, num_epochs=2)
        model.estimator.train()
        original = {name: tensor.detach().clone() for name, tensor in model.estimator.state_dict().items()}
        first = model.predict_proba(self.X)
        np.testing.assert_array_equal(first, model.predict_proba(self.X))
        self.assertTrue(all(not module.training for module in model.estimator.modules() if isinstance(module, nn.Dropout)))
        for name, tensor in model.estimator.state_dict().items():
            torch.testing.assert_close(original[name], tensor, rtol=0, atol=0)
        self.assertEqual(model.predict_proba(self.X[:0]).shape, (0, 3))

    def test_class_prior_fits_training_frequencies_and_argmax_ties(self):
        model = fit_classifier("class_prior", self.X[:6], [0, 0, 0, 1, 1, 2], self.config)
        np.testing.assert_allclose(model.predict_proba(self.X)[0], [.5, 1 / 3, 1 / 6])
        self.assertTrue((model.predict(self.X) == 0).all())
        tied = fit_classifier("class_prior", self.X, self.y, self.config)
        self.assertTrue((tied.predict(self.X) == 0).all())
        missing = fit_classifier("class_prior", self.X[:2], [1, 1], self.config)
        np.testing.assert_array_equal(missing.predict_proba(self.X)[0], [0, 1, 0])

    def test_invalid_training_and_unset_epochs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "fixed num_epochs"):
            fit_classifier("mlp_one_hidden", self.X, self.y, self.config)
        with self.assertRaises(TypeError):
            fit_classifier("logistic", self.X.toarray(), self.y, self.config)
        with self.assertRaises(ValueError):
            fit_classifier("logistic", self.X[:2], [0, 1], self.config)
        with self.assertRaises(ValueError):
            fit_classifier("logistic", self.X, self.y.astype(float), self.config)
        with self.assertRaises(ValueError):
            build_neural_model("mlp_two_hidden", 3, {"hidden_dims": {"mlp_two_hidden": [8]}, "dropout": 0.1})


class PredictionBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.config = settings()
        cls.premises = [f"item {color}" for _ in range(3) for color in ("red", "green", "blue")]
        cls.hypotheses = [f"{color} {color}" for _ in range(3) for color in ("red", "green", "blue")]
        cls.labels = np.tile(np.arange(3), 3)
        cls.encoder, tokens = fit_pair_features(cls.premises, cls.hypotheses, cls.config["tokenizer"], cls.config["bigram_features"])
        cls.matrices = cls.encoder.representations(cls.encoder.count_matrix(tokens))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_all_nine_systems_bundle_their_preprocessing_and_reproduce_probs(self):
        with tempfile.TemporaryDirectory() as temporary:
            for representation in REPRESENTATIONS:
                for kind in CLASSIFIERS:
                    matrix = self.matrices[representation]
                    classifier = fit_classifier(kind, matrix, self.labels, self.config, num_epochs=2)
                    directory = Path(temporary) / representation / kind
                    save_prediction_bundle(directory, classifier, self.encoder, representation, context={"software_check": True})
                    loaded = load_prediction_bundle(directory)
                    np.testing.assert_allclose(
                        classifier.predict_proba(matrix), loaded.predict_proba(self.premises, self.hypotheses),
                        rtol=0, atol=1e-7,
                    )
                    self.assertEqual(loaded.metadata["label_names"], ["entailment", "neutral", "contradiction"])
                    self.assertEqual(loaded.classifier.parameter_count, classifier.parameter_count)
                    self.assertEqual(loaded.predict_proba([], []).shape, (0, 3))

    def test_hypothesis_only_ignores_changed_premise_and_prior_reloads(self):
        representation = "unigram_counts"
        matrix = select_input_features(self.matrices[representation], "hypothesis_only")
        classifier = fit_classifier("logistic", matrix, self.labels, self.config)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "hypothesis"
            save_prediction_bundle(directory, classifier, self.encoder, representation, input_mode="hypothesis_only")
            loaded = load_prediction_bundle(directory)
            first = loaded.predict_proba(self.premises, self.hypotheses)
            np.testing.assert_array_equal(first, loaded.predict_proba(["🦄 totally different"] * len(self.premises), self.hypotheses))
            np.testing.assert_allclose(first, classifier.predict_proba(matrix), atol=1e-7)
            prior = fit_classifier("class_prior", self.matrices[representation], self.labels, self.config)
            save_prediction_bundle(Path(temporary) / "prior", prior, self.encoder, representation)
            restored_prior = load_prediction_bundle(Path(temporary) / "prior")
            np.testing.assert_allclose(restored_prior.predict_proba(self.premises, self.hypotheses), 1 / 3)

    def test_bundle_rejects_wrong_dimensions_overwrite_and_changed_preprocessing(self):
        classifier = fit_classifier("logistic", self.matrices["unigram_counts"], self.labels, self.config)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "bundle"
            with self.assertRaisesRegex(ValueError, "dimensions"):
                save_prediction_bundle(directory, classifier, self.encoder, "unigram_bigram_counts")
            save_prediction_bundle(directory, classifier, self.encoder, "unigram_counts")
            with self.assertRaises(FileExistsError):
                save_prediction_bundle(directory, classifier, self.encoder, "unigram_counts")
            with (directory / "preprocessing/tokenizer.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                load_prediction_bundle(directory)


if __name__ == "__main__":
    unittest.main()
