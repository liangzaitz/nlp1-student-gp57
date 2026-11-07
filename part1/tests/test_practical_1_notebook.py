"""Exercise the actual notebook definitions, without rerunning the SNLI experiment."""
import ast
from collections import Counter
from copy import deepcopy
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict
import nbformat
import numpy as np
import pandas as pd
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'part1/practical_1.ipynb'
CONSTANTS = {'ROOT', 'PART1', 'BASE', 'LABELS', 'REPRESENTATIONS', 'CLASSIFIERS'}


def definitions():
    module = types.ModuleType('notebook_under_test')
    notebook = nbformat.read(NOTEBOOK, 4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not node.level and not (node.module or '').startswith(
                    ('part1', 'snli_', 'project_setup', 'smlp'))
            if isinstance(node, ast.Import):
                assert not any(alias.name.startswith(('part1', 'snli_')) for alias in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {'exec', 'eval', 'compile', '__import__', 'run_search'}
        # Only this external test harness extracts definitions; notebook code is literal Python.
        for node in tree.body:
            include = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
            if isinstance(node, ast.Assign):
                include = all(isinstance(target, ast.Name) and target.id in CONSTANTS
                              for target in node.targets)
            if include:
                statement = ast.Module(body=[node], type_ignores=[])
                exec(compile(statement, f'<cell-{index}>', 'exec'), module.__dict__)
    return module


class TestSharedBPENotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = definitions()
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def config(self):
        config = deepcopy(self.n.BASE)
        config.update(vocab_size=64, bpe_min_frequency=1, bigram_min_frequency=1,
                      max_bigrams=32, epochs=2, batch_size=6,
                      hidden_dims={'mlp_one_hidden': [8], 'mlp_two_hidden': [8, 4]},
                      n_splits=3, cv_seeds=[0])
        return config

    def frame(self, prefix='train', groups=6):
        return pd.DataFrame({
            'premise': [f'{prefix} group {i} CAT' for i in range(groups) for _ in range(3)],
            'hypothesis': [word + ' cat' for _ in range(groups) for word in ('yes', 'maybe', 'no')],
            'label': np.tile(np.arange(3), groups),
            'example_id': [f'{prefix}:{i}' for i in range(groups * 3)],
        }).assign(premise_group=lambda frame: frame.premise)

    def test_development_and_bpe_include_validation_and_exclude_test(self):
        train, validation, test = self.frame(groups=3), self.frame('validation', 2), self.frame('test', 1)
        test.loc[0, 'premise'] = train.loc[0, 'premise']
        test['hypothesis'] = '☃ exclusive test text'
        data = DatasetDict({name: Dataset.from_pandas(frame, preserve_index=False)
                            for name, frame in [('train', train), ('validation', validation), ('test', test)]})
        development = self.n.development_pool(data)
        self.assertEqual(len(development), len(train) + len(validation) - 3)
        self.assertEqual(set(development.source_split), {'train', 'validation'})
        self.assertTrue(set(development.premise).isdisjoint(test.premise))
        tokenizer = self.n.fit_tokenizer(development[['premise', 'hypothesis']], self.config())
        self.assertNotIn('☃', ''.join(tokenizer.get_vocab()))

    def test_features_match_direct_counts_and_never_cross_roles(self):
        n, config = self.n, self.config()
        tokenizer = n.fit_tokenizer(self.frame(), config)
        encoder = n.PairEncoder(tokenizer, n.learn_bigrams(n.encode_pairs(tokenizer, self.frame()), config))
        frame = pd.DataFrame({'premise': ['CAT cat', '', 'cat', '', ''],
                              'hypothesis': ['', 'YES cat', 'cat cat', '', 'cat']})
        inventory = {pair: index for index, pair in enumerate(encoder.bigrams)}
        for representation in n.REPRESENTATIONS:
            width = len(inventory) if representation == 'bigram_counts' else encoder.vocab_size
            expected = np.zeros((len(frame), width * 2), dtype=np.float32)
            for row, texts in enumerate(frame.itertuples(index=False, name=None)):
                for role, text in enumerate(texts):
                    ids = tokenizer.encode(text).ids
                    counts = Counter(zip(ids, ids[1:])) if representation == 'bigram_counts' else Counter(ids)
                    for key, count in counts.items():
                        if representation == 'bigram_counts' and key not in inventory:
                            continue
                        column = inventory[key] if representation == 'bigram_counts' else key
                        expected[row, role * width + column] = 1 if representation == 'indicators' else count
            actual = n.feature_matrix(encoder, n.encode_pairs(tokenizer, frame), representation)
            self.assertTrue(sparse.isspmatrix_csr(actual))
            np.testing.assert_array_equal(actual.toarray(), expected)

    def test_whole_group_sampling_and_fold_separation(self):
        frame = self.frame(groups=12)
        sampled = self.n.sample_groups(frame, 19, seed=0)
        self.assertGreaterEqual(len(sampled), 19)
        self.assertTrue(sampled.groupby('premise_group').size().eq(3).all())
        pd.testing.assert_frame_equal(sampled, self.n.sample_groups(frame, 19, seed=0))
        assignments = self.n.make_assignments(sampled, self.config())
        for fold in range(3):
            heldout = assignments.fold_seed_0.eq(fold)
            self.assertTrue(set(sampled.loc[heldout].premise_group).isdisjoint(
                sampled.loc[~heldout].premise_group))

    def test_fixed_epoch_training_and_deterministic_predictions(self):
        config = self.config()
        features = sparse.csr_matrix(np.tile(np.eye(3, dtype=np.float32), (6, 1)))
        labels = np.tile(np.arange(3), 6)
        for classifier in self.n.CLASSIFIERS:
            model = self.n.fit_model(features, labels, classifier, config)
            repeated = self.n.fit_model(features, labels, classifier, config)
            probabilities = self.n.predict_proba(model, features)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            np.testing.assert_array_equal(probabilities, self.n.predict_proba(repeated, features))
            self.assertEqual(len(model['history']), 0 if classifier == 'logistic' else config['epochs'])

    def test_cv_reuses_bpe_and_covers_all_heldout_examples(self):
        n, config, frame = self.n, self.config(), self.frame()
        tokenizer = n.fit_tokenizer(frame, config)
        original = tokenizer.to_str()
        assignments = n.make_assignments(frame, config)
        with patch.object(n, 'fit_tokenizer', side_effect=AssertionError('BPE must not refit inside CV')):
            results, predictions = n.run_cv(frame, assignments, tokenizer, config)
        self.assertEqual(tokenizer.to_str(), original)
        self.assertEqual(len(results), 3 * 9)
        self.assertEqual(len(predictions), len(frame) * 9)
        self.assertFalse(predictions.duplicated(['row', 'representation', 'classifier']).any())

    def test_selection_uses_accuracy_then_nll(self):
        results = pd.DataFrame([
            {'representation': representation, 'classifier': 'logistic', 'fold': fold,
             'accuracy': accuracy, 'nll': nll, 'macro_f1': accuracy, 'converged': True}
            for representation, accuracy, nll in [('indicators', .6, .8),
                                                  ('unigram_counts', .6, .7), ('bigram_counts', .5, .6)]
            for fold in [0, 1]
        ])
        self.assertEqual(self.n.summarise_cv(results).iloc[0].representation, 'unigram_counts')

    def test_metrics_and_posterior_direction(self):
        metrics = self.n.score(np.arange(3), np.eye(3))
        self.assertEqual(metrics, {'accuracy': 1, 'nll': 0, 'macro_f1': 1})
        equal = self.n.correlated_posterior(np.full(5, .6), np.full(5, .6))
        self.assertEqual(equal['P_equivalent'], 1)
        improvement = self.n.correlated_posterior(np.full(5, .5), np.full(5, .7))
        self.assertEqual(improvement['P_B_better'], 1)
        self.assertAlmostEqual(improvement['difference_B_minus_A'], .2)

    def test_posterior_scale_and_probabilities_match_tutorial_library(self):
        from baycomp import CorrelatedTTest

        a = np.array([.60, .61, .59, .60, .62])
        b = np.array([.61, .63, .60, .62, .63])
        mean, scale_squared, degrees = CorrelatedTTest.compute_statistics(a, b, runs=1)
        for rope in [.0025, .005, .01]:
            result = self.n.correlated_posterior(a, b, n_splits=5, rope=rope)
            self.assertAlmostEqual(result['difference_B_minus_A'], mean)
            self.assertAlmostEqual(result['student_scale'] ** 2, scale_squared)
            self.assertEqual(result['degrees_of_freedom'], degrees)
            np.testing.assert_allclose(
                [result['P_A_better'], result['P_equivalent'], result['P_B_better']],
                CorrelatedTTest.probs(a, b, rope=rope, runs=1), atol=1e-12,
            )


if __name__ == '__main__':
    unittest.main()
