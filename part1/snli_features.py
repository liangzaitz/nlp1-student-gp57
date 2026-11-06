"""Training-only BPE and sparse sentence-pair features for Project 1.

Run ``python -m part1.snli_features --seed 0 --fold 0`` after Steps 2-3.
Each fold gets its own fitted tokenizer, bigram inventory, tokenization cache,
and count matrices. Other folds are prepared on demand before their model fits.
"""

import argparse
from array import array
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import chain
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from datasets import load_from_disk
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
import tokenizers
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Metaspace

from .experiment import train_tokenizer
from .project_setup import DEFAULT_CONFIG, prepare_workspace
from .snli_evaluation import ROW_COLUMNS, prepare_evaluation_protocol


FEATURE_VERSION = "v1"
REPRESENTATIONS = ("indicators", "unigram_counts", "unigram_bigram_counts")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TokenizedPairs:
    """Compact ragged token arrays; offsets mark sentence boundaries per role."""

    premise_ids: np.ndarray
    premise_offsets: np.ndarray
    hypothesis_ids: np.ndarray
    hypothesis_offsets: np.ndarray

    def __post_init__(self):
        for ids, offsets in self.roles():
            if ids.ndim != 1 or offsets.ndim != 1 or not len(offsets):
                raise ValueError("Token IDs and offsets must be flat arrays with an initial offset.")
            if ids.dtype != np.int32 or offsets.dtype != np.int64:
                raise ValueError("Expected int32 token IDs and int64 sentence offsets.")
            if offsets[0] != 0 or offsets[-1] != len(ids) or np.any(np.diff(offsets) < 0):
                raise ValueError("Invalid sentence offsets.")
        if len(self.premise_offsets) != len(self.hypothesis_offsets):
            raise ValueError("Premise/hypothesis sentence counts differ.")

    def __len__(self) -> int:
        return len(self.premise_offsets) - 1

    def roles(self):
        return (
            (self.premise_ids, self.premise_offsets),
            (self.hypothesis_ids, self.hypothesis_offsets),
        )

    def pair(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= row < len(self):
            raise IndexError(row)
        return (
            self.premise_ids[self.premise_offsets[row]:self.premise_offsets[row + 1]],
            self.hypothesis_ids[self.hypothesis_offsets[row]:self.hypothesis_offsets[row + 1]],
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, premise_ids=self.premise_ids, premise_offsets=self.premise_offsets,
            hypothesis_ids=self.hypothesis_ids, hypothesis_offsets=self.hypothesis_offsets,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TokenizedPairs":
        with np.load(path, allow_pickle=False) as data:
            return cls(**{name: data[name] for name in cls.__dataclass_fields__})


def tokenize_pairs(
    tokenizer: Tokenizer, premises: Sequence[str], hypotheses: Sequence[str],
    *, batch_size: int = 2048,
) -> TokenizedPairs:
    """Encode roles independently in bounded batches, without padding/truncation.

    Blank inference sentences map to an empty sequence/zero role block. Nonblank
    input is passed unchanged; whitespace handling belongs to Metaspace.
    """
    if len(premises) != len(hypotheses) or batch_size < 1:
        raise ValueError("Equal role lengths and a positive batch size are required.")
    buffers = []
    for sentences in (premises, hypotheses):
        ids, offsets = array("i"), array("q", [0])
        for start in range(0, len(sentences), batch_size):
            texts = list(sentences[start:start + batch_size])
            if any(not isinstance(text, str) for text in texts):
                raise TypeError("Each premise/hypothesis must be a string.")
            texts = [text if text.strip() else "" for text in texts]
            for encoding in tokenizer.encode_batch(texts, add_special_tokens=False):
                ids.extend(encoding.ids)
                offsets.append(len(ids))
        buffers.extend((np.asarray(ids, dtype=np.int32), np.asarray(offsets, dtype=np.int64)))
    return TokenizedPairs(*buffers)


def learn_bigrams(
    tokens: TokenizedPairs, min_frequency: int, max_features: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Rank training-only within-sentence BPE bigrams by occurrence counts.

    Ties use ascending token-ID pairs. Count both roles in one shared inventory,
    never creating a bigram across sentences, rows, or role boundaries.
    """
    if min_frequency < 1 or max_features < 0:
        raise ValueError("Bigram minimum frequency must be positive; cap must be nonnegative.")
    counts = Counter()
    for ids, offsets in tokens.roles():
        for start, end in zip(offsets[:-1], offsets[1:]):
            sentence = ids[start:end]
            counts.update(zip(sentence[:-1], sentence[1:]))
    selected = sorted(
        ((pair, count) for pair, count in counts.items() if count >= min_frequency),
        key=lambda item: (-item[1], item[0]),
    )[:max_features]
    return [tuple(map(int, pair)) for pair, _ in selected], [int(count) for _, count in selected]


class PairFeatureEncoder:
    """A fitted tokenizer/inventory; transformation never learns new coordinates.

    Unigram spaces: [P unigrams; H unigrams]. Extended count space:
    [P unigrams; P bigrams; H unigrams; H bigrams]. All values are raw counts
    or binary indicators, stored as float32 CSR without dense intermediate rows.
    """

    def __init__(
        self, tokenizer: Tokenizer, bigrams: list[tuple[int, int]],
        bigram_counts: list[int], unk_token: str = "[UNK]",
    ):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.bigrams = tuple(tuple(pair) for pair in bigrams)
        self.bigram_counts = tuple(bigram_counts)
        self.unk_token = unk_token
        if len(self.bigrams) != len(self.bigram_counts) or len(set(self.bigrams)) != len(self.bigrams):
            raise ValueError("Bigram inventory/counts must align and contain no duplicates.")
        if any(len(pair) != 2 or any(not 0 <= token < self.vocab_size for token in pair) for pair in self.bigrams):
            raise ValueError("Bigram IDs must refer to the fitted vocabulary.")
        if any(count < 1 for count in self.bigram_counts) or tokenizer.token_to_id(unk_token) is None:
            raise ValueError("Expected positive training bigram counts and an unknown token.")
        self._bigram_ids = {pair: self.vocab_size + i for i, pair in enumerate(self.bigrams)}

    def num_features(self, representation: str) -> int:
        if representation not in REPRESENTATIONS:
            raise ValueError(f"Unknown representation: {representation}")
        return 2 * (self.vocab_size + (len(self.bigrams) if representation == "unigram_bigram_counts" else 0))

    def count_matrix(self, tokens: TokenizedPairs) -> sparse.csr_matrix:
        """Construct the full unigram+bigram count matrix directly in CSR."""
        for ids, _ in tokens.roles():
            if len(ids) and (ids.min() < 0 or ids.max() >= self.vocab_size):
                raise ValueError("Token IDs are outside this encoder's fitted vocabulary.")
        width = self.num_features("unigram_bigram_counts") // 2
        values, indices, indptr = array("f"), array("i"), array("q", [0])
        for row in range(len(tokens)):
            coordinates = {}
            for offset, sentence in zip((0, width), tokens.pair(row)):
                coordinates.update((offset + int(token), count) for token, count in Counter(sentence).items())
                for pair, count in Counter(zip(sentence[:-1], sentence[1:])).items():
                    feature = self._bigram_ids.get(pair)
                    if feature is not None:
                        coordinates[offset + feature] = count
            for coordinate, count in sorted(coordinates.items()):
                indices.append(coordinate)
                values.append(count)
            indptr.append(len(indices))
        return sparse.csr_matrix((
            np.asarray(values, dtype=np.float32), np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ), shape=(len(tokens), 2 * width))

    def representations(self, counts: sparse.csr_matrix) -> dict[str, sparse.csr_matrix]:
        if counts.shape[1] != self.num_features("unigram_bigram_counts"):
            raise ValueError("Count matrix has the wrong feature dimension.")
        width = counts.shape[1] // 2
        unigrams = sparse.hstack((
            counts[:, :self.vocab_size], counts[:, width:width + self.vocab_size],
        ), format="csr", dtype=np.float32)
        indicators = unigrams.copy()
        indicators.data.fill(1)
        return {"indicators": indicators, "unigram_counts": unigrams, "unigram_bigram_counts": counts}

    def transform(
        self, premises: Sequence[str], hypotheses: Sequence[str], representation: str,
    ) -> sparse.csr_matrix:
        self.num_features(representation)  # Reject an unknown space before encoding.
        tokens = tokenize_pairs(self.tokenizer, premises, hypotheses)
        return self.representations(self.count_matrix(tokens))[representation]

    def feature_name(self, coordinate: int, representation: str) -> str:
        dimension = self.num_features(representation)
        if not 0 <= coordinate < dimension:
            raise IndexError(coordinate)
        role, local = divmod(coordinate, dimension // 2)
        if local < self.vocab_size:
            name = repr(self.tokenizer.id_to_token(local))
        else:
            name = repr(tuple(self.tokenizer.id_to_token(token) for token in self.bigrams[local - self.vocab_size]))
        return f"{'P' if role == 0 else 'H'}: {name}"

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "encoder.json").write_text(json.dumps({
            "feature_version": FEATURE_VERSION, "unk_token": self.unk_token,
            "bigrams": self.bigrams, "bigram_counts": self.bigram_counts,
            "bigram_tie_break": "frequency_descending_then_token_id_pair_ascending",
            "role_layout": "P_unigrams,P_bigrams,H_unigrams,H_bigrams",
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "PairFeatureEncoder":
        directory = Path(directory)
        metadata = json.loads((directory / "encoder.json").read_text(encoding="utf-8"))
        if metadata["feature_version"] != FEATURE_VERSION:
            raise ValueError("Unsupported saved feature version.")
        return cls(
            Tokenizer.from_file(str(directory / "tokenizer.json")),
            metadata["bigrams"], metadata["bigram_counts"], metadata["unk_token"],
        )


def fit_pair_features(
    premises: Sequence[str], hypotheses: Sequence[str],
    tokenizer_config: dict[str, Any], bigram_config: dict[str, Any],
) -> tuple[PairFeatureEncoder, TokenizedPairs]:
    """Fit only on the supplied training sentences; also return their encoding."""
    if len(premises) != len(hypotheses) or not len(premises):
        raise ValueError("Fitting requires nonempty, equally sized training roles.")
    if any(not isinstance(text, str) or not text.strip() for text in chain(premises, hypotheses)):
        raise ValueError("Fitting requires cleaned, nonblank training sentences.")
    if tokenizer_config["algorithm"] != "BPE" or tokenizer_config["pretokenizer"] != "Metaspace":
        raise ValueError("This implementation uses the tutorial's BPE/Metaspace helper.")
    if tokenizer_config.get("normalization", "none") != "none":
        raise ValueError("This feature comparison preserves case and punctuation without normalization.")
    if tokenizer_config["vocab_size"] < 1 or tokenizer_config["min_frequency"] < 1:
        raise ValueError("BPE vocabulary target and merge minimum frequency must be positive.")
    unk = tokenizer_config.get("unk_token", "[UNK]")
    tokenizer = train_tokenizer(
        chain.from_iterable(zip(premises, hypotheses)),
        vocab_size=tokenizer_config["vocab_size"], min_frequency=tokenizer_config["min_frequency"],
        pretokenizer=Metaspace(replacement="▁", prepend_scheme="always", split=True), unk_token=unk,
    )
    tokens = tokenize_pairs(tokenizer, premises, hypotheses)
    bigrams, counts = learn_bigrams(tokens, **bigram_config)
    return PairFeatureEncoder(tokenizer, bigrams, counts, unk), tokens


def matrix_summary(matrices: dict[str, dict[str, sparse.csr_matrix]]) -> pd.DataFrame:
    rows = []
    for partition, representations in matrices.items():
        for name, matrix in representations.items():
            rows.append({
                "partition": partition, "representation": name,
                "rows": matrix.shape[0], "columns": matrix.shape[1], "nonzeros": matrix.nnz,
                "dtype": str(matrix.dtype),
                "csr_bytes": matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
                "estimated_dense_bytes": matrix.shape[0] * matrix.shape[1] * matrix.dtype.itemsize,
            })
    return pd.DataFrame(rows)


def prepare_fold_features(
    config_path: str | Path = DEFAULT_CONFIG, *, seed: int = 0, fold: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fit/cache one outer fold, or verify/reuse its existing fitted artifacts.

    Canonical extended counts are cached once; unigram counts and indicators
    derive from those same coordinates/tokenizations for all model families.
    Official validation/test sentences are never tokenized here.
    """
    started = perf_counter()
    workspace = prepare_workspace(config_path)
    config, paths = workspace["config"], workspace["paths"]
    protocol = prepare_evaluation_protocol(config_path)
    if seed not in config["seeds"]["cv_splits"] or not 0 <= fold < config["cross_validation"]["num_folds"]:
        raise ValueError("Choose a seed/fold from the saved evaluation protocol.")
    if config["representations"] != list(REPRESENTATIONS):
        raise ValueError(f"Expected the three planned representations: {REPRESENTATIONS}")
    rows = protocol["assignments"][ROW_COLUMNS].copy()
    rows["partition"] = np.where(protocol["assignments"][f"fold_seed_{seed}"].eq(fold), "heldout", "train")
    partitions = {name: rows.loc[rows["partition"].eq(name)].reset_index(drop=True) for name in ("train", "heldout")}
    specification = {
        "feature_version": FEATURE_VERSION, "protocol_id": protocol["manifest"]["protocol_id"],
        "seed": seed, "fold": fold, "tokenizer": config["tokenizer"],
        "bigram_features": config["bigram_features"], "representations": list(REPRESENTATIONS),
        "tokenizers_version": tokenizers.__version__, "scipy_version": scipy.__version__,
        "implementation_sha256": _sha256(Path(__file__)),
        "tutorial_helper_sha256": _sha256(Path(__file__).with_name("experiment.py")),
    }
    feature_id = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:16]
    directory = paths["features"] / specification["protocol_id"] / f"seed_{seed}_fold_{fold}" / feature_id
    manifest_path = directory / "manifest.json"
    reused = directory.exists()
    if reused:
        if not manifest_path.is_file():
            raise ValueError(f"Incomplete feature cache: {directory}. Existing files were not overwritten.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["specification"] != specification:
            raise ValueError("Saved features do not match this fold/configuration.")
        for name, digest in manifest["artifact_sha256"].items():
            path = directory / name
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"Saved feature artifact changed or is missing: {path}")
        if not pd.read_parquet(directory / "rows.parquet").equals(rows):
            raise ValueError("Cached feature rows do not match the saved fold order.")
        encoder = PairFeatureEncoder.load(directory)
        tokenized = {name: TokenizedPairs.load(directory / f"{name}_tokens.npz") for name in partitions}
        counts = {name: sparse.load_npz(directory / f"{name}_counts.npz").tocsr() for name in partitions}
        if verbose:
            print(f"Reused features for seed {seed}, fold {fold}.", flush=True)
    else:
        if verbose:
            print(f"Fitting BPE/inventory on {len(partitions['train']):,} training pairs (seed {seed}, fold {fold})...", flush=True)
        source = load_from_disk(protocol["manifest"]["prepared_dataset"])["train"]
        frame = source.select(partitions["train"]["clean_index"].tolist()).select_columns(["premise", "hypothesis"]).to_pandas()
        fit_started = perf_counter()
        encoder, train_tokens = fit_pair_features(
            frame["premise"].tolist(), frame["hypothesis"].tolist(), config["tokenizer"], config["bigram_features"],
        )
        fitted_at = perf_counter()
        frame = source.select(partitions["heldout"]["clean_index"].tolist()).select_columns(["premise", "hypothesis"]).to_pandas()
        tokenized = {
            "train": train_tokens,
            "heldout": tokenize_pairs(encoder.tokenizer, frame["premise"].tolist(), frame["hypothesis"].tolist()),
        }
        del frame
        if verbose:
            print("Building float32 CSR counts from the shared tokenization...", flush=True)
        counts = {name: encoder.count_matrix(tokens) for name, tokens in tokenized.items()}
        directory.mkdir(parents=True)
        encoder.save(directory)
        rows.to_parquet(directory / "rows.parquet", index=False)
        for name in partitions:
            tokenized[name].save(directory / f"{name}_tokens.npz")
            sparse.save_npz(directory / f"{name}_counts.npz", counts[name])
        manifest = {
            "stage": "sentence_pair_features", "feature_id": feature_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "specification": specification,
            "creation_config_sha256": workspace["manifest"]["config_sha256"],
            "repository": workspace["manifest"]["repository"],
            "actual_vocab_size": encoder.vocab_size, "bigram_types": len(encoder.bigrams),
            "training_rows": len(partitions["train"]), "heldout_rows": len(partitions["heldout"]),
            "normalization": "No case folding, punctuation removal, stemming, or negation rewrite; Metaspace handles whitespace.",
            "fit_corpus": "Training premise then hypothesis per row, including repeated text; both roles share one vocabulary/inventory.",
            "bigram_selection": "Within-sentence occurrence counts in both training roles; frequency descending, token-ID-pair ascending.",
            "feature_layout": "P_unigrams,P_bigrams,H_unigrams,H_bigrams; drop bigram blocks for unigram spaces.",
            "fit_seconds": fitted_at - fit_started, "preparation_seconds": perf_counter() - started,
            "test_usage": "No official validation/test text is tokenized or used to fit these features.",
            "classifier_trained": False,
            "artifact_sha256": {path.name: _sha256(path) for path in sorted(directory.iterdir()) if path.is_file()},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for name in partitions:
        if len(tokenized[name]) != len(partitions[name]) or counts[name].shape != (
            len(partitions[name]), encoder.num_features("unigram_bigram_counts"),
        ) or counts[name].dtype != np.float32:
            raise ValueError(f"{name}: tokenization/counts do not align with this fold.")
    matrices = {name: encoder.representations(matrix) for name, matrix in counts.items()}
    return {
        "encoder": encoder, "tokens": tokenized, "matrices": matrices, "rows": partitions,
        "summary": matrix_summary(matrices), "manifest": manifest,
        "manifest_path": manifest_path, "feature_dir": directory, "reused_features": reused,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    arguments = parser.parse_args()
    features = prepare_fold_features(arguments.config, seed=arguments.seed, fold=arguments.fold, verbose=True)
    print(features["summary"].to_string(index=False))
    print("\nVocabulary/bigram types:", features["encoder"].vocab_size, len(features["encoder"].bigrams))
    print("Reused saved features:", features["reused_features"])
    print("Feature manifest:", features["manifest_path"])
