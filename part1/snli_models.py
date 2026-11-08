"""Matrix-based classifiers, metrics, and complete prediction bundles for SNLI.

The one-hidden-layer network reuses the tutorial SparseMLP unchanged. Its
two-hidden-layer extension adds another affine/ReLU/dropout stage. Fixed-epoch
training never receives outer held-out labels or selects a checkpoint on them.
Run ``python -m part1.snli_models`` for synthetic software checks, not SNLI CV.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any, Callable, Sequence
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch import nn

from .project_setup import DEFAULT_CONFIG, prepare_workspace, set_random_seeds
from .smlp import SparseMLP
from .snli_data import LABEL_NAMES
from .snli_features import PairFeatureEncoder, REPRESENTATIONS, _sha256, fit_pair_features


MODEL_VERSION = "v1"
CLASSIFIERS = ("logistic", "mlp_one_hidden", "mlp_two_hidden")
CLASS_IDS = np.arange(len(LABEL_NAMES))


def _matrix(X) -> sparse.csr_matrix:
    if not sparse.isspmatrix_csr(X):
        raise TypeError("Expected a SciPy CSR matrix; full feature matrices must stay sparse.")
    if not X.shape[1] or not np.isfinite(X.data).all():
        raise ValueError("Expected a positive input dimension and finite feature values.")
    return X.astype(np.float32, copy=False)


def _labels(y, rows: int, *, require_all: bool = False) -> np.ndarray:
    values = np.asarray(y)
    if values.ndim != 1 or len(values) != rows or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("Expected one integer gold label per row.")
    if not np.isin(values, CLASS_IDS).all() or (require_all and set(values.tolist()) != set(CLASS_IDS.tolist())):
        raise ValueError("Training needs classes 0, 1, 2; evaluation labels must lie in that set.")
    return values.astype(np.int64, copy=False)


def validate_probabilities(probabilities, rows: int | None = None) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABEL_NAMES) or (rows is not None and len(values) != rows):
        raise ValueError("Expected probability columns in label order [0, 1, 2].")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-7):
        raise ValueError("Each probability row must sum to one.")
    return values


def score_probabilities(y, probabilities) -> dict[str, Any]:
    """Apply Step 3's fixed class order, argmax ties, and reporting conventions."""
    probabilities = validate_probabilities(probabilities)
    if not len(probabilities):
        raise ValueError("Cannot score an empty evaluation set.")
    labels = _labels(y, len(probabilities))
    predicted = probabilities.argmax(axis=1)  # First maximum: smallest class ID.
    epsilon = np.finfo(np.float64).eps
    nll = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], epsilon, 1 - epsilon)).mean()
    return {
        "accuracy": float(accuracy_score(labels, predicted)), "nll": float(nll),
        "macro_f1": float(f1_score(labels, predicted, labels=CLASS_IDS, average="macro", zero_division=0)),
        "per_class": classification_report(
            labels, predicted, labels=CLASS_IDS, target_names=list(LABEL_NAMES),
            zero_division=0, output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=CLASS_IDS).tolist(),
        "num_examples": len(labels),
    }


def select_input_features(X: sparse.csr_matrix, input_mode: str = "pair") -> sparse.csr_matrix:
    X = _matrix(X)
    if input_mode == "pair":
        return X
    if input_mode == "hypothesis_only" and X.shape[1] % 2 == 0:
        return X[:, X.shape[1] // 2:]
    raise ValueError("Expected pair or hypothesis_only mode with equal sentence-role blocks.")


def scipy_batch_to_torch(X: sparse.csr_matrix, device: str = "cpu") -> torch.Tensor:
    """Convert one sparse batch to COO; the feature matrix is never densified."""
    coo = X.tocoo(copy=False)
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64, copy=False))
    values = torch.from_numpy(np.asarray(coo.data, dtype=np.float32))
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, check_invariants=True,
    ).coalesce().to(device)


class PairMLP(SparseMLP):
    """Tutorial sparse first layer, with one or two ReLU/dropout hidden stages."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float):
        if input_dim < 1 or len(hidden_dims) not in (1, 2) or any(width < 1 for width in hidden_dims):
            raise ValueError("Expected a positive input dimension and one/two positive hidden widths.")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must lie in [0, 1).")
        super().__init__(input_dim, hidden_dims[0], len(LABEL_NAMES), dropout)
        if len(hidden_dims) == 2:
            self.output = nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dims[1], len(LABEL_NAMES)), nn.LogSoftmax(dim=-1),
            )


def build_neural_model(kind: str, input_dim: int, settings: dict[str, Any]) -> PairMLP:
    if kind not in CLASSIFIERS[1:]:
        raise ValueError(f"Unknown neural architecture: {kind}")
    dimensions = settings["hidden_dims"][kind]
    if len(dimensions) != (1 if kind == "mlp_one_hidden" else 2):
        raise ValueError("Architecture name and hidden-layer count disagree.")
    return PairMLP(input_dim, dimensions, settings["dropout"])


@dataclass
class FittedClassifier:
    kind: str
    estimator: Any
    input_dim: int
    settings: dict[str, Any]
    history: list[dict[str, Any]]
    fit_info: dict[str, Any]

    @property
    def parameter_count(self) -> int:
        if self.kind == "logistic":
            return int(self.estimator.coef_.size + self.estimator.intercept_.size)
        if self.kind == "class_prior":
            return len(self.estimator)  # Stored prior probabilities; they sum to one.
        return sum(parameter.numel() for parameter in self.estimator.parameters())

    def predict_log_proba(self, X: sparse.csr_matrix, *, batch_size: int = 1024) -> np.ndarray:
        X = _matrix(X)
        if X.shape[1] != self.input_dim or batch_size < 1:
            raise ValueError("Input dimension must match training, and batch size must be positive.")
        if not X.shape[0]:
            return np.empty((0, len(LABEL_NAMES)), dtype=np.float64)
        if self.kind == "logistic":
            if not np.array_equal(self.estimator.classes_, CLASS_IDS):
                raise ValueError("Classifier class order is not [0, 1, 2].")
            return np.asarray(self.estimator.predict_log_proba(X), dtype=np.float64)
        if self.kind == "class_prior":
            with np.errstate(divide="ignore"):
                return np.tile(np.log(self.estimator), (X.shape[0], 1))
        model = self.estimator
        model.eval()  # Ordinary evaluation disables both hidden-layer dropouts.
        device = str(model.weight.device)
        batches = []
        with torch.no_grad():
            for start in range(0, X.shape[0], batch_size):
                output = model(scipy_batch_to_torch(X[start:start + batch_size], device))
                batches.append(output.cpu().numpy().astype(np.float64))
        return np.concatenate(batches)

    def predict_proba(self, X: sparse.csr_matrix, *, batch_size: int = 1024) -> np.ndarray:
        return validate_probabilities(np.exp(self.predict_log_proba(X, batch_size=batch_size)), X.shape[0])

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def fit_classifier(
    kind: str, X: sparse.csr_matrix, y, config: dict[str, Any], *,
    seed: int | None = None, num_epochs: int | None = None,
    epoch_observer: Callable[[FittedClassifier], None] | None = None,
) -> FittedClassifier:
    """Fit with a temporary configured CPU thread count; restore it afterward.

    An observer may log independent pilot-validation curves. Its return value
    cannot alter training length or select a checkpoint. Reported CV must not
    pass outer held-out scores through an observer for model selection.
    """
    original_threads = torch.get_num_threads()
    threads = config["neural_training"].get("num_threads") if kind in CLASSIFIERS[1:] else None
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        raise ValueError("num_threads must be a positive integer.")
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        return _fit_classifier(kind, X, y, config, seed=seed, num_epochs=num_epochs, epoch_observer=epoch_observer)
    finally:
        if threads is not None:
            torch.set_num_threads(original_threads)


def _fit_classifier(
    kind: str, X: sparse.csr_matrix, y, config: dict[str, Any], *,
    seed: int | None = None, num_epochs: int | None = None,
    epoch_observer: Callable[[FittedClassifier], None] | None = None,
) -> FittedClassifier:
    """Fit from CSR using training labels only; save the final fixed-epoch state.

    ``num_epochs`` can explicitly override the unset pilot value for software
    checks. Reported runs must supply the frozen pilot-selected value. There is
    no outer validation argument and no checkpoint selection on held-out scores.
    """
    X = _matrix(X)
    if not X.shape[0] or kind not in (*CLASSIFIERS, "class_prior"):
        raise ValueError("Choose a supported classifier and nonempty training matrix.")
    labels = _labels(y, X.shape[0], require_all=kind != "class_prior")
    seed = config["seeds"]["model"] if seed is None else seed
    set_random_seeds(seed)
    started = perf_counter()
    common = {"seed": seed, "training_rows": X.shape[0], "class_order": CLASS_IDS.tolist()}
    if kind == "class_prior":
        counts = np.bincount(labels, minlength=len(LABEL_NAMES))
        return FittedClassifier(kind, counts / counts.sum(), X.shape[1], {}, [], {
            **common, "class_counts": counts.tolist(), "fit_seconds": perf_counter() - started,
        })
    if kind == "logistic":
        settings = dict(config["logistic_training"])
        if settings["solver"] != "saga":
            raise ValueError("Use the tutorial's sparse-friendly SAGA solver for this comparison.")
        model = LogisticRegression(**settings, l1_ratio=0.0, fit_intercept=True, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, labels)
        convergence = [str(warning.message) for warning in caught if issubclass(warning.category, ConvergenceWarning)]
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn(str(warning.message), warning.category)
        return FittedClassifier(kind, model, X.shape[1], settings, [], {
            **common, "regularization": "L2 (l1_ratio=0); C is inverse strength",
            "converged": not convergence, "convergence_warnings": convergence,
            "iterations": model.n_iter_.tolist(), "fit_seconds": perf_counter() - started,
        })

    settings = dict(config["neural_training"])
    epochs = settings["num_epochs"] if num_epochs is None else num_epochs
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("Set positive fixed num_epochs after the pilot, or explicitly override for a software check.")
    if settings["optimizer"] != "AdamW" or settings["batch_size"] < 1:
        raise ValueError("Expected AdamW and a positive sparse batch size.")
    settings["num_epochs"] = epochs
    device = config["device"]
    model = build_neural_model(kind, X.shape[1], settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
    )
    criterion = nn.NLLLoss()
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(X.shape[0])
        total_loss, total_correct = 0.0, 0
        epoch_started = perf_counter()
        for start in range(0, len(order), settings["batch_size"]):
            indices = order[start:start + settings["batch_size"]]
            features = scipy_batch_to_torch(X[indices], device)
            target = torch.as_tensor(labels[indices], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            log_probabilities = model(features)
            loss = criterion(log_probabilities, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite neural training loss.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
            total_correct += int((log_probabilities.argmax(dim=1) == target).sum().item())
        history.append({
            "epoch": epoch, "train_nll_online": total_loss / X.shape[0],
            "train_accuracy_online": total_correct / X.shape[0],
            "seconds": perf_counter() - epoch_started,
        })
        if epoch_observer is not None:
            epoch_observer(FittedClassifier(kind, model, X.shape[1], settings, history, {
                **common, "epochs_completed": epoch, "checkpoint_policy": "epoch_observer_view",
            }))
    model.eval()
    return FittedClassifier(kind, model, X.shape[1], settings, history, {
        **common, "device": device, "batch_order_seed": seed,
        "checkpoint_policy": "last_fixed_epoch", "epochs_completed": epochs,
        "weight_decay_scope": "all parameters, including biases",
        "fit_seconds": perf_counter() - started, "torch_cpu_threads": torch.get_num_threads(),
    })


@dataclass
class PredictionBundle:
    classifier: FittedClassifier
    encoder: PairFeatureEncoder
    representation: str
    input_mode: str
    metadata: dict[str, Any]

    def predict_proba(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        if len(premises) != len(hypotheses):
            raise ValueError("Premise/hypothesis lengths differ.")
        # The ablation ignores per-example premise content, keeping shared training preprocessing.
        effective_premises = [""] * len(hypotheses) if self.input_mode == "hypothesis_only" else premises
        matrix = self.encoder.transform(effective_premises, hypotheses, self.representation)
        return self.classifier.predict_proba(select_input_features(matrix, self.input_mode))

    def predict(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        return self.predict_proba(premises, hypotheses).argmax(axis=1)


def save_prediction_bundle(
    directory: str | Path, classifier: FittedClassifier, encoder: PairFeatureEncoder,
    representation: str, *, input_mode: str = "pair", context: dict[str, Any] | None = None,
) -> Path:
    expected_dim = encoder.num_features(representation)
    if input_mode == "hypothesis_only":
        expected_dim //= 2
    elif input_mode != "pair":
        raise ValueError("Unknown input mode.")
    if classifier.input_dim != expected_dim:
        raise ValueError("Classifier and saved preprocessing dimensions disagree.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    encoder.save(directory / "preprocessing")
    if classifier.kind == "logistic":
        joblib.dump(classifier.estimator, directory / "model.joblib")
    elif classifier.kind == "class_prior":
        np.save(directory / "prior.npy", classifier.estimator, allow_pickle=False)
    else:
        state = {name: tensor.detach().cpu() for name, tensor in classifier.estimator.state_dict().items()}
        torch.save(state, directory / "model.pt")
    metadata = {
        "model_version": MODEL_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": classifier.kind, "representation": representation, "input_mode": input_mode,
        "input_dim": classifier.input_dim, "label_names": list(LABEL_NAMES),
        "class_order": CLASS_IDS.tolist(), "tie_break": "smallest_label_id",
        "settings": classifier.settings, "fit_info": classifier.fit_info,
        "history": classifier.history, "parameter_count": classifier.parameter_count,
        "context": context or {}, "torch_version": torch.__version__, "sklearn_version": sklearn.__version__,
        "implementation_sha256": _sha256(Path(__file__)),
        "tutorial_mlp_sha256": _sha256(Path(__file__).with_name("smlp.py")),
        "artifact_sha256": {
            str(path.relative_to(directory)): _sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
        },
    }
    path = directory / "bundle.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def load_prediction_bundle(directory: str | Path, *, device: str = "cpu") -> PredictionBundle:
    directory = Path(directory)
    metadata = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if metadata["model_version"] != MODEL_VERSION or metadata["label_names"] != list(LABEL_NAMES) or metadata["class_order"] != CLASS_IDS.tolist():
        raise ValueError("Unsupported bundle version or label mapping.")
    if metadata["input_mode"] not in ("pair", "hypothesis_only"):
        raise ValueError("Unsupported saved input mode.")
    for name, digest in metadata["artifact_sha256"].items():
        path = directory / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"Saved model/preprocessing artifact changed or is missing: {path}")
    kind = metadata["kind"]
    if kind == "logistic":
        estimator = joblib.load(directory / "model.joblib")
    elif kind == "class_prior":
        estimator = np.load(directory / "prior.npy", allow_pickle=False)
    elif kind in CLASSIFIERS[1:]:
        estimator = build_neural_model(kind, metadata["input_dim"], metadata["settings"])
        estimator.load_state_dict(torch.load(directory / "model.pt", map_location=device, weights_only=True))
        estimator = estimator.to(device)
        estimator.eval()
    else:
        raise ValueError(f"Unknown saved classifier: {kind}")
    classifier = FittedClassifier(kind, estimator, metadata["input_dim"], metadata["settings"], metadata["history"], metadata["fit_info"])
    encoder = PairFeatureEncoder.load(directory / "preprocessing")
    expected = encoder.num_features(metadata["representation"]) // (2 if metadata["input_mode"] == "hypothesis_only" else 1)
    if expected != classifier.input_dim:
        raise ValueError("Saved model/preprocessing dimensions disagree.")
    return PredictionBundle(classifier, encoder, metadata["representation"], metadata["input_mode"], metadata)


def run_model_smoke(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Exercise all nine systems + baselines on synthetic, arbitrarily labelled text.

    No SNLI examples/scores or epoch choices are involved. Two epochs merely
    exercise neural training/reload; the experiment's num_epochs stays unset.
    """
    workspace = prepare_workspace(config_path)
    config, paths = workspace["config"], workspace["paths"]
    destination = paths["checkpoints"] / "implementation_smoke"
    destination.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run_", dir=destination))
    train_p = [f"item {color}" for _ in range(4) for color in ("red", "green", "blue")]
    train_h = [f"{color} {color}" for _ in range(4) for color in ("red", "green", "blue")]
    train_y = np.tile(CLASS_IDS, 4)
    eval_p, eval_h = ["item red", "item green", "item blue"], ["red", "green", "blue"]
    encoder, tokens = fit_pair_features(train_p, train_h, config["tokenizer"], config["bigram_features"])
    matrices = encoder.representations(encoder.count_matrix(tokens))
    checks, bundles = [], {}
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # Small mechanical checks; restore the user's setting below.
    try:
        systems = [(representation, kind, "pair") for representation in REPRESENTATIONS for kind in CLASSIFIERS]
        diagnostic = config["diagnostics"]["hypothesis_only"]
        systems += [
            ("unigram_counts", "class_prior", "pair"),
            (diagnostic["representation"], diagnostic["classifier"], diagnostic["input_mode"]),
        ]
        for representation, kind, mode in systems:
            X = select_input_features(matrices[representation], mode)
            classifier = fit_classifier(kind, X, train_y, config, num_epochs=2)
            full_eval = encoder.transform(eval_p, eval_h, representation)
            probabilities = classifier.predict_proba(select_input_features(full_eval, mode))
            scores = score_probabilities(CLASS_IDS, probabilities)
            name = f"{representation}__{kind}__{mode}"
            directory = run_dir / name
            save_prediction_bundle(directory, classifier, encoder, representation, input_mode=mode, context={
                "purpose": "synthetic_software_check_not_snli", "config_sha256": workspace["manifest"]["config_sha256"],
                "labels": "Arbitrary red/green/blue software-check targets, not NLI annotations.",
            })
            restored = load_prediction_bundle(directory)
            reloaded = restored.predict_proba(eval_p, eval_h)
            difference = float(np.max(np.abs(probabilities - reloaded)))
            if not np.allclose(probabilities, reloaded, rtol=0, atol=1e-7):
                raise AssertionError("Saved bundle did not reproduce its probabilities.")
            checks.append({
                "representation": representation, "classifier": kind, "input_mode": mode,
                "input_dim": classifier.input_dim, "parameters": classifier.parameter_count,
                "probabilities_valid": True, "finite_nll": bool(np.isfinite(scores["nll"])),
                "reload_max_abs_difference": difference,
                "logistic_converged": classifier.fit_info.get("converged"),
                "epochs": classifier.fit_info.get("epochs_completed"),
            })
            bundles[name] = directory
    finally:
        torch.set_num_threads(original_threads)
    table = pd.DataFrame(checks)
    table.to_csv(run_dir / "checks.csv", index=False)
    (run_dir / "smoke.json").write_text(json.dumps({
        "purpose": "synthetic_software_check_not_snli", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(train_y), "evaluation_rows": len(eval_h), "neural_epochs_for_check_only": 2,
        "dataset_revision": config["dataset"]["revision"], "repository": workspace["manifest"]["repository"],
        "snli_rows_used": 0, "reported_cv_results": False, "checks": checks,
    }, indent=2) + "\n", encoding="utf-8")
    return {"checks": table, "run_dir": run_dir, "bundles": bundles}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    smoke = run_model_smoke(arguments.config)
    print(smoke["checks"].to_string(index=False))
    print("\nSynthetic checks only; no SNLI model comparison.")
    print("Saved prediction bundles:", smoke["run_dir"])
