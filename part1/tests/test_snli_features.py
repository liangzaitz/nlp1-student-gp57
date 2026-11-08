"""Offline feature arithmetic, fit-boundary, persistence, and cache checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from part1.project_setup import DEFAULT_CONFIG
from part1.snli_data import CLEANING_VERSION
from part1.snli_features import (
    PairFeatureEncoder, REPRESENTATIONS, TokenizedPairs,
    fit_pair_features, learn_bigrams, prepare_fold_features, tokenize_pairs,
)
from part1.tests.test_snli_evaluation import example_splits


def word_fixture():
    # A fixed tiny vocabulary makes manual feature arithmetic independent of BPE.
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "red": 1, "blue": 2, "green": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


class SparseFeatureTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = word_fixture()
        self.premises = ["red red blue", "green"]
        self.hypotheses = ["blue red", "blue"]
        self.tokens = tokenize_pairs(self.tokenizer, self.premises, self.hypotheses, batch_size=1)
        pairs, counts = learn_bigrams(self.tokens, min_frequency=1, max_features=10)
        self.encoder = PairFeatureEncoder(self.tokenizer, pairs, counts)

    def test_manual_counts_indicators_dimensions_and_sparse_dtype(self):
        matrices = self.encoder.representations(self.encoder.count_matrix(self.tokens))
        np.testing.assert_array_equal(
            matrices["unigram_counts"].toarray(),
            [[0, 2, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0, 1, 0]],
        )
        np.testing.assert_array_equal(
            matrices["indicators"].toarray(),
            [[0, 1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 0, 1, 0]],
        )
        # B=3: only P bigrams followed by H bigrams; single-token roles are zero.
        np.testing.assert_array_equal(
            matrices["bigram_counts"].toarray(),
            [[1, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]],
        )
        for name, matrix in matrices.items():
            self.assertTrue(sparse.isspmatrix_csr(matrix))
            self.assertEqual(matrix.dtype, np.float32)
            self.assertEqual(matrix.shape[1], self.encoder.num_features(name))
            self.assertTrue(matrix.has_canonical_format)

    def test_bigram_frequency_cap_ties_and_sentence_boundaries(self):
        pairs, counts = learn_bigrams(self.tokens, min_frequency=1, max_features=2)
        self.assertEqual(pairs, [(1, 1), (1, 2)])
        self.assertEqual(counts, [1, 1])
        self.assertNotIn((2, 2), self.encoder.bigrams)  # P end -> H start.
        self.assertNotIn((2, 3), self.encoder.bigrams)  # Across premise rows.
        self.assertEqual(learn_bigrams(self.tokens, min_frequency=2, max_features=5), ([], []))
        zero = PairFeatureEncoder(self.tokenizer, [], [])
        matrix = zero.transform(self.premises, self.hypotheses, "bigram_counts")
        self.assertEqual(matrix.shape, (2, 0))

    def test_bigram_only_counts_names_roles_and_omitted_pairs(self):
        # Retain repeated-pair counts, but never fall back to unigram values.
        original = self.encoder.transform(["red red red blue"], ["blue red"], "bigram_counts")
        np.testing.assert_array_equal(original.toarray(), [[2, 1, 0, 0, 0, 1]])
        self.assertEqual(self.encoder.feature_name(0, "bigram_counts"), "P: ('red', 'red')")
        self.assertEqual(self.encoder.feature_name(5, "bigram_counts"), "H: ('blue', 'red')")
        swapped = self.encoder.transform(["blue red"], ["red red red blue"], "bigram_counts")
        np.testing.assert_array_equal(original[:, :3].toarray(), swapped[:, 3:].toarray())
        np.testing.assert_array_equal(original[:, 3:].toarray(), swapped[:, :3].toarray())
        empty = self.encoder.transform(["red", "blue", "red green", ""], [""] * 4, "bigram_counts")
        self.assertEqual(empty.shape, (4, 6))
        self.assertEqual(empty.nnz, 0)  # Known unigrams and an omitted pair contribute nothing.
        with self.assertRaisesRegex(ValueError, "Unknown representation"):
            self.encoder.transform(["red"], ["blue"], "unigram_bigram_counts")

    def test_order_collision_and_sentence_role_swap(self):
        tokens = tokenize_pairs(self.tokenizer, ["red blue green", "blue red green"], ["green", "green"])
        pairs, counts = learn_bigrams(tokens, 1, 20)
        encoder = PairFeatureEncoder(self.tokenizer, pairs, counts)
        matrices = encoder.representations(encoder.count_matrix(tokens))
        self.assertEqual((matrices["unigram_counts"][0] != matrices["unigram_counts"][1]).nnz, 0)
        self.assertGreater((matrices["bigram_counts"][0] != matrices["bigram_counts"][1]).nnz, 0)
        original = encoder.transform(["red red"], ["blue"], "unigram_counts")
        swapped = encoder.transform(["blue"], ["red red"], "unigram_counts")
        np.testing.assert_array_equal(original[:, :4].toarray(), swapped[:, 4:].toarray())
        np.testing.assert_array_equal(original[:, 4:].toarray(), swapped[:, :4].toarray())

    def test_empty_unknown_malformed_inputs_and_unrecognized_space(self):
        matrix = self.encoder.transform(["", "  ", "unseen"], ["", "", ""], "unigram_counts")
        self.assertEqual(matrix[:2].nnz, 0)
        self.assertEqual(matrix[2, 0], 1)
        self.assertEqual(matrix[2, 4:].nnz, 0)
        for representation in REPRESENTATIONS:
            self.assertEqual(self.encoder.transform([], [], representation).shape, (0, self.encoder.num_features(representation)))
        with self.assertRaises(ValueError):
            tokenize_pairs(self.tokenizer, ["red"], [])
        with self.assertRaises(TypeError):
            tokenize_pairs(self.tokenizer, [None], ["red"])
        with self.assertRaises(ValueError):
            self.encoder.transform(["red"], ["blue"], "tfidf")
        bad = tokenize_pairs(self.tokenizer, ["red"], ["blue"])
        bad.premise_ids[0] = 999
        with self.assertRaisesRegex(ValueError, "outside"):
            self.encoder.count_matrix(bad)

    def test_batched_and_serialized_tokens_and_encoder_round_trip(self):
        other = tokenize_pairs(self.tokenizer, self.premises, self.hypotheses, batch_size=5)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            self.tokens.save(path / "tokens.npz")
            loaded = TokenizedPairs.load(path / "tokens.npz")
            for first, second, third in zip(self.tokens.roles(), other.roles(), loaded.roles()):
                for a, b, c in zip(first, second, third):
                    np.testing.assert_array_equal(a, b)
                    np.testing.assert_array_equal(a, c)
            self.encoder.save(path)
            encoder = PairFeatureEncoder.load(path)
            for representation in REPRESENTATIONS:
                original = self.encoder.transform(self.premises, self.hypotheses, representation)
                restored = encoder.transform(self.premises, self.hypotheses, representation)
                self.assertEqual((original != restored).nnz, 0)
                self.assertEqual(
                    self.encoder.feature_name(original.indices[0], representation),
                    encoder.feature_name(original.indices[0], representation),
                )


class BpeFitBoundaryTests(unittest.TestCase):
    def test_real_bpe_preserves_text_and_never_learns_from_transform(self):
        tokenizer_config = {"algorithm": "BPE", "pretokenizer": "Metaspace", "vocab_size": 100, "min_frequency": 2}
        encoder, _ = fit_pair_features(
            ["Red DOG! dog", "A red dog is not a cat."] * 3,
            ["The dog is here.", "The cat is not here!"] * 3,
            tokenizer_config, {"min_frequency": 2, "max_features": 10},
        )
        before = encoder.tokenizer.to_str()
        vocabulary = encoder.tokenizer.get_vocab()
        self.assertIsNone(encoder.tokenizer.normalizer)
        self.assertTrue(any("R" in token for token in vocabulary))
        self.assertTrue(any("!" in token for token in vocabulary))
        self.assertIn(encoder.tokenizer.token_to_id("[UNK]"), encoder.tokenizer.encode("🦄").ids)
        self.assertNotIn("🦄", vocabulary)
        for representation in REPRESENTATIONS:
            matrix = encoder.transform(["🦄 new heldout text"], ["not known"], representation)
            self.assertEqual(matrix.shape[1], encoder.num_features(representation))
        self.assertEqual(before, encoder.tokenizer.to_str())
        self.assertEqual(vocabulary, encoder.tokenizer.get_vocab())

    def test_actual_vocabulary_size_and_invalid_fit_inputs(self):
        config = {"algorithm": "BPE", "pretokenizer": "Metaspace", "vocab_size": 5, "min_frequency": 2}
        encoder, _ = fit_pair_features(["abcdefghijk"], ["lmnopqrstuv"], config, {"min_frequency": 2, "max_features": 0})
        self.assertGreater(encoder.vocab_size, 5)  # The observed alphabet already exceeds the target.
        self.assertEqual(encoder.num_features("indicators"), 2 * encoder.vocab_size)
        for premises, hypotheses in (([], []), ([""], ["ok"]), (["ok"], [])):
            with self.assertRaises(ValueError):
                fit_pair_features(premises, hypotheses, config, {"min_frequency": 2, "max_features": 0})


class FoldFeatureCacheTests(unittest.TestCase):
    def test_fold_specific_cache_reuse_config_separation_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(json.loads(DEFAULT_CONFIG.read_text()))
            config["paths"] = {"data_dir": str(root / "data"), "output_dir": str(root / "results")}
            config["cross_validation"]["num_folds"] = 3
            config["seeds"]["cv_splits"] = [0, 3]
            config["tokenizer"]["vocab_size"] = 64
            config["bigram_features"]["max_features"] = 5
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            prepared = root / "data/snli" / config["dataset"]["revision"] / f"cleaned-{CLEANING_VERSION}"
            dataset = example_splits()
            dataset.save_to_disk(str(prepared))

            first = prepare_fold_features(config_path, seed=0, fold=0)
            count_path = first["feature_dir"] / "train_counts.npz"
            initial_mtime = count_path.stat().st_mtime_ns
            config["neural_training"]["learning_rate"] = 0.002
            config_path.write_text(json.dumps(config))
            second = prepare_fold_features(config_path, seed=0, fold=0)
            self.assertFalse(first["reused_features"])
            self.assertTrue(second["reused_features"])
            self.assertEqual(first["feature_dir"], second["feature_dir"])
            self.assertEqual(initial_mtime, count_path.stat().st_mtime_ns)
            for partition in ("train", "heldout"):
                rows = second["rows"][partition]
                source = dataset["train"].select(rows["clean_index"].tolist())
                np.testing.assert_array_equal(source["example_id"], rows["example_id"])
                tokens = tokenize_pairs(second["encoder"].tokenizer, source["premise"], source["hypothesis"])
                reconstructed = second["encoder"].representations(second["encoder"].count_matrix(tokens))
                for representation in REPRESENTATIONS:
                    self.assertEqual((reconstructed[representation] != second["matrices"][partition][representation]).nnz, 0)
                    self.assertEqual((first["matrices"][partition][representation] != second["matrices"][partition][representation]).nnz, 0)
            alternate = prepare_fold_features(config_path, seed=3, fold=1)
            self.assertNotEqual(first["feature_dir"], alternate["feature_dir"])
            config["bigram_features"]["max_features"] = 6
            config_path.write_text(json.dumps(config))
            changed = prepare_fold_features(config_path, seed=0, fold=0)
            self.assertNotEqual(first["feature_dir"], changed["feature_dir"])
            self.assertEqual(first["manifest"]["specification"]["protocol_id"], changed["manifest"]["specification"]["protocol_id"])
            config["bigram_features"]["max_features"] = 5
            config_path.write_text(json.dumps(config))
            with (first["feature_dir"] / "tokenizer.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                prepare_fold_features(config_path, seed=0, fold=0)


if __name__ == "__main__":
    unittest.main()
