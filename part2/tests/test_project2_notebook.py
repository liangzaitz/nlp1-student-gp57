"""Offline checks of notebook definitions; no datasets, downloads or heavy runs.

Run: .venv/bin/python -m unittest discover -s part2/tests -v
"""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / 'part2/part2_prelim/project2.ipynb'


def notebook_definitions():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    # Load course encoders directly: package __init__ also requires NumPyro.
    spec = importlib.util.spec_from_file_location('course_clslib', ROOT / 'part2/clslib.py')
    course = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(course)
    env = {'__name__': __name__, 'clslib': course}
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    imports = [n for n in ast.parse(code_cells[0].source).body
               if isinstance(n, (ast.Import, ast.ImportFrom))
               and not (isinstance(n, ast.ImportFrom) and n.module == 'part2')]
    exec(compile(ast.Module(body=imports, type_ignores=[]), str(NOTEBOOK), 'exec'), env)
    for cell in code_cells:
        constants = [n for n in ast.parse(cell.source).body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in {'BASE', 'LABELS'} for t in n.targets)]
        exec(compile(ast.Module(body=constants, type_ignores=[]), str(NOTEBOOK), 'exec'), env)
    env['DEVICE'] = torch.device('cpu')
    for cell in code_cells:
        nodes = [n for n in ast.parse(cell.source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(NOTEBOOK), 'exec'), env)
    return env


class Project2NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.env = notebook_definitions()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'toy.txt'
            path.write_text('dog 1 0 0 0\ncat 0 1 0 0\nnot 0 0 1 0\nruns 0 0 0 1\n')
            cls.emb = cls.env['clslib'].PretrainedEmbeddings(str(path), 4)
        cls.env['embeddings'] = {4: cls.emb}
        cls.pairs = [(np.array([2,3]), np.array([4])),
                     (np.array([4]), np.array([5,3])),
                     (np.array([5,2]), np.array([2]))]*2
        cls.labels = np.array([0,1,2]*2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, encoder='lstm'):
        return {'encoder': encoder, 'head': 'mlp2', 'dim': 4,
                'hidden': 5, 'dropout': 0., 'learning_rate': .001}

    def test_notebook_valid_compiles_and_is_unexecuted(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                self.assertGreater(i, 0)
                self.assertEqual(notebook.cells[i-1].cell_type, 'markdown')
                compile(cell.source, f'{NOTEBOOK}:cell{i}', 'exec')
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

    def test_explanatory_text_keeps_reference_hidden(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                text = cell.source.lower()
                for reference in ['practical 1', 'practical_1', 'project 1', 'project1']:
                    self.assertNotIn(reference, text)

    def test_complete_groups_and_disjoint_folds(self):
        frame = pd.DataFrame({'premise': np.repeat([f'p{i}' for i in range(30)], 3),
                              'label': np.tile([0,1,2], 30)})
        sample = self.env['sample_groups'](frame, 22, 0)
        self.assertEqual(len(sample), 24)
        self.assertTrue(sample.groupby('premise').size().eq(3).all())
        pd.testing.assert_frame_equal(sample, self.env['sample_groups'](frame, 22, 0))
        splitter = self.env['StratifiedGroupKFold'](n_splits=5, shuffle=True, random_state=0)
        for a,b in splitter.split(frame, frame.label, frame.premise):
            self.assertTrue(set(frame.iloc[a].premise).isdisjoint(frame.iloc[b].premise))

    def test_unknown_empty_sentences_and_mean_pooling(self):
        ts = {'premise': [['dog','cat'], []], 'hypothesis': [['not'], ['missing']]}
        pairs = self.env['encode_pairs'](ts, self.emb)
        self.assertEqual(pairs[1][0].tolist(), [self.emb.vocab.default_index])
        features = self.env['pooled_pairs'](pairs, self.emb)
        np.testing.assert_allclose(features[0,:4], [.5,.5,0,0])
        np.testing.assert_allclose(features[1,:4], self.emb.embed_word('<unk>'))
        rows = [(torch.as_tensor(p), torch.as_tensor(h), 0) for p,h in pairs]
        p,h,_ = self.env['collate_pairs'](rows)
        encoder = self.env['clslib'].EmbedAndPoolEncoder(torch.from_numpy(self.emb.vectors))
        with torch.no_grad():
            torch.testing.assert_close(encoder(p), torch.from_numpy(features[:,:4]))
            torch.testing.assert_close(encoder(h), torch.from_numpy(features[:,4:]))

    def test_encoders_ignore_padding_and_preserve_roles(self):
        for family in ['pool','lstm']:
            model = self.env['PairClassifier'](self.config(family), self.emb).eval()
            p,h = torch.tensor([[2,3]]), torch.tensor([[4,5]])
            with torch.no_grad():
                output = model(p,h)
                padded = model(torch.tensor([[2,3,0,0]]), torch.tensor([[4,5,0]]))
                reverse = model(h,p)
            torch.testing.assert_close(output, padded)
            torch.testing.assert_close(output.exp().sum(1), torch.ones(1))
            self.assertFalse(torch.allclose(output, reverse))
            self.assertFalse(model.encoder.embedding.weight.requires_grad)
            if family == 'lstm':
                self.assertEqual(model.encoder.lstm.num_layers, 2)

    def test_tiny_fit_freezes_vectors_and_weights_reload(self):
        config = self.config()
        make_loader = self.env['make_loader']
        fitting = make_loader(self.pairs, self.labels, np.arange(3), shuffle=True)
        heldout = make_loader(self.pairs, self.labels, np.arange(3,6))
        before = self.emb.vectors.copy()
        model, history, epoch = self.env['fit_model'](config, fitting, heldout, epochs=1)
        probabilities, gold = self.env['predict'](model, heldout)
        np.testing.assert_array_equal(before, self.emb.vectors)
        self.assertEqual(epoch, 1)
        self.assertEqual(len(history), 1)
        weights = self.env['model_weights'](model)
        self.assertNotIn('encoder.embedding.weight', weights)
        restored = self.env['PairClassifier'](config, self.emb).eval()
        missing, extra = restored.load_state_dict(weights, strict=False)
        self.assertEqual(missing, ['encoder.embedding.weight'])
        self.assertEqual(extra, [])
        reloaded, _ = self.env['predict'](restored, heldout)
        np.testing.assert_allclose(probabilities, reloaded)
        self.assertTrue(np.isfinite(list(self.env['score'](gold, probabilities).values())).all())

    def test_dev_epoch_selection_starts_at_five(self):
        calls = iter([.95,.9,.85,.8,.7,.6])
        def validation(model, loader):
            gold = np.array([0,1,2])
            p = next(calls)
            probabilities = np.full((3,3), (1-p)/2)
            probabilities[np.arange(3),gold] = p
            return probabilities, gold
        loader = self.env['make_loader'](self.pairs, self.labels)
        with patch.dict(self.env, predict=validation):
            _, _, epoch = self.env['fit_model'](self.config('pool'), loader, loader,
                                               epochs=6, select_epoch=True)
        self.assertEqual(epoch, 5)

    def test_cv_and_dev_search_use_pilot_epoch_budget(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        calls = [node for cell in notebook.cells if cell.cell_type == 'code'
                 for node in ast.walk(ast.parse(cell.source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == 'fit_model']
        tuning = next(call for call in calls if any(
            keyword.arg == 'select_epoch' and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True for keyword in call.keywords))
        self.assertEqual(ast.unparse(tuning.args[3]), 'TRAIN_EPOCHS')
        self.assertEqual(sum(len(call.args) > 3 and isinstance(call.args[3], ast.Name)
                             and call.args[3].id == 'TRAIN_EPOCHS' for call in calls), 2)

    def test_reduced_budget_retains_all_eight_designs(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = next(c.source for c in notebook.cells
                      if c.cell_type == 'code' and c.source.startswith('CANDIDATES = {}'))
        scope = dict(self.env)
        # Only construct design dictionaries; do not run the pilot.
        exec(source.split('MODEL_NAMES =', 1)[0], scope)
        designs, base = scope['CANDIDATES'], scope['BASE']
        self.assertEqual(base['training_rows'], 60000)
        self.assertEqual(base['cv_seeds'], [0, 3])
        self.assertEqual(base['min_epochs'], 5)
        self.assertEqual(base['max_epochs'], 10)
        self.assertEqual(len(designs), 8)
        self.assertEqual(sum(c['encoder'] == 'pool' for c in designs.values()), 6)
        self.assertEqual(len(designs)*base['n_splits']*len(base['cv_seeds']), 80)
        cv_source = next(c.source for c in notebook.cells if c.cell_type == 'code'
                         and c.source.startswith('cv_rows, oof_rows'))
        fit_seed = next(n for n in ast.walk(ast.parse(cv_source))
                        if isinstance(n, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == 'fit_seed' for t in n.targets))
        seeds = []
        for seed in base['cv_seeds']:
            scope['seed'] = seed
            exec(compile(ast.Module(body=[fit_seed], type_ignores=[]), 'cv_seed', 'exec'), scope)
            seeds.append(scope['fit_seed'])
        self.assertEqual(seeds, [0, 3])

    def test_reduced_search_changes_one_parameter_per_trial(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = next(c.source for c in notebook.cells
                      if c.cell_type == 'code' and c.source.startswith('tuning_rows, tuned'))
        family_loop = next(n for n in ast.parse(source).body if isinstance(n, ast.For))
        option_nodes = []
        for node in family_loop.body:
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == 'pairs' for t in node.targets):
                break
            option_nodes.append(node)
        counts = []
        for family, expected in [('pool', 6), ('lstm', 8)]:
            base = {'encoder': family, 'head': 'mlp2', 'dim': 300,
                    'hidden': 300, 'dropout': .1, 'learning_rate': .001}
            scope = dict(self.env, family=family, CANDIDATES={'chosen': base},
                         selected={family: 'chosen'})
            exec(compile(ast.Module(body=option_nodes, type_ignores=[]), 'search', 'exec'), scope)
            options = scope['options']
            self.assertEqual(len(options), expected)
            self.assertEqual(options['incumbent'], base)
            counts.append(len(options))
            for name, config in options.items():
                if name != 'incumbent':
                    self.assertEqual(sum(config[k] != base[k] for k in base), 1)
            for parameter, values in scope['search'].items():
                self.assertTrue(set(values) <= {c[parameter] for c in options.values()})
        self.assertEqual(sum(counts), 14)
        self.assertEqual(2 + 80 + sum(counts) + 2, 98)

    def test_chunked_neighbours_match_dense_search(self):
        rng = np.random.default_rng(1)
        queries, references = rng.normal(size=(7,4)), rng.normal(size=(11,4))
        best, positions = self.env['nearest_hypotheses'](queries, references, 3, 4)
        dense = ((queries/np.linalg.norm(queries,axis=1,keepdims=True)) @
                 (references/np.linalg.norm(references,axis=1,keepdims=True)).T)
        np.testing.assert_allclose(best, dense.max(1))
        np.testing.assert_array_equal(positions, dense.argmax(1))
        with self.assertRaises(ValueError):
            self.env['nearest_hypotheses'](queries, references[:0])

    def test_correlated_posterior_direction_and_rope(self):
        result = self.env['correlated_posterior'](np.zeros(5), np.full(5,.02))
        self.assertAlmostEqual(result['mean'], .02)
        self.assertEqual(result['P_B'], 1.)
        a = np.array([.5,.6,.7,.8,.9])
        b = a + np.array([.01,-.01,.02,.03,-.02])
        forward = self.env['correlated_posterior'](a,b)
        reverse = self.env['correlated_posterior'](b,a)
        self.assertAlmostEqual(forward['P_A'], reverse['P_B'])
        self.assertAlmostEqual(forward['P_A']+forward['P_equal']+forward['P_B'], 1.)
        expected_scale = np.sqrt((1/len(a)+1/4)*(b-a).var(ddof=1))
        self.assertAlmostEqual(forward['scale'], expected_scale)


if __name__ == '__main__':
    unittest.main()
