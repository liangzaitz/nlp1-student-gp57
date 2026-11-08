"""Step 8: deterministic Bayesian comparisons of checked, paired CV accuracies.

Run ``python -m part1.snli_analysis`` after completing Step 7. Neither the
runner nor the read-only notebook loader trains models or scores test data.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from itertools import combinations
from pathlib import Path
import tempfile

import baycomp.single
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from .analysis import BenavoliCorrelatedT
from .project_setup import REPO_ROOT
from .snli_experiments import load_cv_report, planned_systems
from .snli_pilot import FROZEN_CONFIG


ANALYSIS_POLICY = REPO_ROOT / "part1/configs/snli_analysis.json"
KEYS = ["repetition", "seed", "fold"]
TABLES = ("paired_accuracy", "contrasts", "paired_differences", "posteriors", "rope_sensitivity")
FAMILIES = ("representation", "architecture", "class_prior", "hypothesis_ablation")
LABELS = {"indicators": "Indicators", "unigram_counts": "Unigrams", "bigram_counts": "Bigrams",
          "logistic": "LR", "mlp_one_hidden": "MLP1", "mlp_two_hidden": "MLP2"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def planned_contrasts(config):
    """Choose comparisons from design factors, never from observed rankings.

    A/B order follows the frozen lists. All 18 one-factor contrasts are primary;
    nine comparisons to the prior and one fixed input ablation provide context.
    """
    systems = planned_systems(config)
    rows = []

    def add(family, a, b, label):
        rows.append({"contrast_id": f"{family}_{sum(r['family'] == family for r in rows) + 1:02d}",
                     "family": family, "primary": family in FAMILIES[:2],
                     "system_a": a, "system_b": b, "label": label})

    for classifier in config["classifiers"]:
        for a, b in combinations(config["representations"], 2):
            add("representation", f"{a}__{classifier}__pair", f"{b}__{classifier}__pair",
                f"{LABELS[classifier]}: {LABELS[a]} → {LABELS[b]}")
    for representation in config["representations"]:
        for a, b in combinations(config["classifiers"], 2):
            add("architecture", f"{representation}__{a}__pair", f"{representation}__{b}__pair",
                f"{LABELS[representation]}: {LABELS[a]} → {LABELS[b]}")
    prior = next(s["system_id"] for s in systems if s["classifier"] == "class_prior")
    for system in systems:
        if system["primary"]:
            add("class_prior", prior, system["system_id"],
                f"Prior → {LABELS[system['representation']]} {LABELS[system['classifier']]}")
    diagnostic = next(s for s in systems if s["input_mode"] == "hypothesis_only")
    add("hypothesis_ablation", f"{diagnostic['representation']}__{diagnostic['classifier']}__pair",
        diagnostic["system_id"], "Unigram LR: pair → hypothesis only")
    return pd.DataFrame(rows)


def paired_scores(metrics, config):
    """Validate the exact grid and align each system by partition identifiers."""
    systems = planned_systems(config)
    ids = [s["system_id"] for s in systems]
    seeds, folds = config["seeds"]["cv_splits"], config["cross_validation"]["num_folds"]
    required = {*KEYS, "system_id", "accuracy", "training_rows", "heldout_rows", "feature_id",
                "run_id", "protocol_id", "config_sha256", "model_seed",
                "representation", "classifier", "input_mode", "primary"}
    if not required.issubset(metrics.columns):
        raise ValueError("Missing CV pairing/provenance columns.")
    if metrics[list(required)].isna().any().any():
        raise ValueError("Missing values in CV pairing/provenance columns.")
    expected = pd.MultiIndex.from_tuples(
        [(r, seed, fold, system) for r, seed in enumerate(seeds) for fold in range(folds) for system in ids],
        names=[*KEYS, "system_id"],
    )
    actual = pd.MultiIndex.from_frame(metrics[[*KEYS, "system_id"]])
    if actual.has_duplicates or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError("Expected a complete paired grid with unique repetition/seed/fold/system keys.")
    accuracy = metrics["accuracy"].to_numpy(dtype=float)
    if not np.isfinite(accuracy).all() or np.any((accuracy < 0) | (accuracy > 1)):
        raise ValueError("Accuracies must be finite fractions in [0, 1], not percentages.")
    for field in ("run_id", "protocol_id", "config_sha256", "model_seed"):
        if metrics[field].nunique() != 1:
            raise ValueError(f"Mixed CV provenance: {field}.")
    if not metrics["model_seed"].eq(config["seeds"]["model"]).all():
        raise ValueError("Model seed differs from the frozen design.")
    for system in systems:
        subset = metrics.loc[metrics["system_id"].eq(system["system_id"])]
        if any(not subset[key].eq(system[key]).all() for key in ("representation", "classifier", "input_mode", "primary")):
            raise ValueError("System metadata disagrees with its planned identity.")
    for field in ("feature_id", "training_rows", "heldout_rows"):
        if not metrics.groupby(KEYS)[field].nunique().eq(1).all():
            raise ValueError(f"Systems do not share paired fold {field}.")
    partitions = metrics.drop_duplicates(KEYS)
    if (partitions[["training_rows", "heldout_rows"]] <= 0).any().any():
        raise ValueError("Each fold needs positive training and held-out sizes.")
    totals = partitions["training_rows"] + partitions["heldout_rows"]
    if totals.nunique() != 1 or not partitions.groupby("repetition")["heldout_rows"].sum().eq(totals.iloc[0]).all():
        raise ValueError("Each repetition must cover the same CV pool once.")
    index = pd.MultiIndex.from_tuples([(r, s, f) for r, s in enumerate(seeds) for f in range(folds)], names=KEYS)
    return metrics.pivot(index=KEYS, columns="system_id", values="accuracy").reindex(index=index, columns=ids).reset_index()


def correlated_posterior(a, b, *, runs, rope, credible_mass=0.95):
    """Analytic posterior for mean accuracy B−A, with repeated-CV correction.

    The tutorial wrapper's ``std`` is Student scale, not distribution SD.
    Constant differences are a point mass; do not call a zero-scale t CDF.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 4:
        raise ValueError("Need matching one-dimensional score arrays with at least four folds.")
    if not isinstance(runs, int) or isinstance(runs, bool) or runs < 1 or len(a) % runs or len(a) // runs < 2:
        raise ValueError("runs must specify complete repetitions with at least two folds each.")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any((a < 0) | (a > 1) | (b < 0) | (b > 1)):
        raise ValueError("Scores must be finite accuracy fractions in [0, 1].")
    if not np.isfinite(rope) or rope <= 0 or not 0 < credible_mass < 1:
        raise ValueError("Need positive finite ROPE and credible mass strictly between zero and one.")
    posterior = BenavoliCorrelatedT(a, b, runs=runs, names=["A", "B"])
    mean, scale, df = float(posterior.mean_diff), float(posterior.std), int(posterior.df)
    if scale == 0:
        pa, pb = float(mean < -rope), float(mean > rope)
        pe, low, high = 1 - pa - pb, mean, mean
    else:
        distribution = t(df, loc=mean, scale=scale)
        pa, pb = float(distribution.cdf(-rope)), float(distribution.sf(rope))
        # SF differences preserve tiny interval probabilities in the right tail.
        pe = float(distribution.sf(-rope) - distribution.sf(rope) if -rope > mean
                   else distribution.cdf(rope) - distribution.cdf(-rope))
        low, high = map(float, distribution.interval(credible_mass))
        if not np.allclose([pa, pe, pb], posterior.probs(rope), rtol=0, atol=1e-12):
            raise ValueError("Analytic region probabilities disagree with the supplied tutorial helper.")
    probabilities = np.array([pa, pe, pb])
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Invalid posterior region probabilities.")
    n, k = len(a), len(a) // runs
    return {"n_scores": n, "runs": runs, "folds_per_run": k, "df": df,
            "rope_half_width": float(rope), "credible_mass": float(credible_mass),
            "mean_difference": mean, "mean_difference_pp": mean * 100,
            "sample_difference_variance": float(np.var(b - a, ddof=1)),
            "correction_factor": 1 / n + 1 / (k - 1),
            "student_scale": scale, "posterior_std": scale * np.sqrt(df / (df - 2)),
            "ci_low": low, "ci_high": high, "ci_low_pp": low * 100, "ci_high_pp": high * 100,
            "p_a_better": pa, "p_equivalent": pe, "p_b_better": pb,
            "point_mass": scale == 0}


def _plot_posterior(row, path):
    rope, mean, scale = row["rope_half_width"] * 100, row["mean_difference_pp"], row["student_scale"] * 100
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axvspan(-rope, rope, color="teal", alpha=.18, label=f"ROPE ±{rope:g} pp")
    if row["point_mass"]:
        ax.axvline(mean, color="navy", label="Point mass")
        ax.set_xlim(min(-rope * 2, mean - rope), max(rope * 2, mean + rope))
    else:
        distribution = t(row["df"], loc=mean, scale=scale)
        lo, hi = distribution.interval(.999)
        x = np.linspace(min(lo, -rope * 1.5), max(hi, rope * 1.5), 1400)
        ax.plot(x, distribution.pdf(x), color="navy", label="Posterior density")
        ax.set_xlim(x[0], x[-1])
    ax.axvline(0, color="gray", linewidth=.7)
    ax.plot([row["ci_low_pp"], row["ci_high_pp"]], [.035, .035], color="black", lw=3,
            transform=ax.get_xaxis_transform(), label=f"{row['credible_mass']:.0%} credible interval")
    ax.set(title=row["label"], xlabel="Mean accuracy difference B − A (percentage points)", ylabel="Density per percentage point")
    fig.text(.5, .01, f"A: {row['system_a']}\nB: {row['system_b']}\n"
             f"P(A better)={row['p_a_better']:.2%}   P(equivalent)={row['p_equivalent']:.2%}   "
             f"P(B better)={row['p_b_better']:.2%}", ha="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, .15, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_overview(posteriors, path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    for ax, family in zip(axes.flat, FAMILIES):
        subset = posteriors.loc[posteriors["family"].eq(family)].reset_index(drop=True)
        y = np.arange(len(subset))
        rope = subset["rope_half_width"].iloc[0] * 100
        ax.axvspan(-rope, rope, color="teal", alpha=.18)
        ax.axvline(0, color="gray", lw=.7)
        mean = subset["mean_difference_pp"].to_numpy()
        ax.errorbar(mean, y, xerr=np.stack([mean - subset["ci_low_pp"], subset["ci_high_pp"] - mean]),
                    fmt="o", markersize=4, capsize=3, color="navy")
        ax.set_yticks(y, subset["label"], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(family.replace("_", " ").title())
        ax.set_xlabel("B − A accuracy (percentage points)")
        ax.grid(axis="x", alpha=.2)
        if len(subset) == 1:
            ax.set_ylim(.5, -.5)
    mass = posteriors["credible_mass"].iloc[0]
    rope = posteriors["rope_half_width"].iloc[0] * 100
    fig.suptitle(f"Paired Bayesian comparisons: mean and {mass:.0%} credible interval\n"
                 f"A → B follows each row label; shaded ROPE ±{rope:g} pp; panels have separate x scales", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _context(config_path, policy_path):
    config_path, policy_path = REPO_ROOT / config_path, REPO_ROOT / policy_path
    config, policy = _read(config_path), _read(policy_path)
    if (policy["method"] != "tutorial_BenavoliCorrelatedT" or policy["difference"] != "B_minus_A"
            or policy["contrast_policy"] != "all_one_factor_pairs_plus_prior_and_fixed_hypothesis_ablation"):
        raise ValueError("Unsupported analysis policy.")
    mass = policy["credible_mass"]
    widths = policy["sensitivity_rope_half_widths"]
    rope = config["evaluation"]["rope_half_width"]
    if not 0 < mass < 1 or rope <= 0 or not np.isfinite(rope) or not widths or any(not np.isfinite(w) or w <= 0 for w in widths):
        raise ValueError("Invalid credible mass or ROPE policy.")
    report = load_cv_report(config_path)
    metrics = report["tables"]["fold_metrics"]
    paired = paired_scores(metrics, config)
    expected = paired.set_index(KEYS)
    saved_paired = report["tables"]["paired_accuracy"]
    existing = saved_paired.set_index(KEYS)
    if existing.index.has_duplicates or len(existing) != len(expected) or set(existing.index) != set(expected.index):
        raise ValueError("Saved paired accuracy table has different partition keys.")
    existing = existing.reindex(expected.index)
    if (set(existing.columns) != set(expected.columns)
            or not np.allclose(existing[expected.columns], expected, rtol=0, atol=1e-14)):
        raise ValueError("Saved paired accuracy table disagrees with fold metrics.")
    for key, value in (("run_id", report["manifest"]["run_id"]),
                       ("protocol_id", report["manifest"]["specification"]["protocol_id"]),
                       ("config_sha256", _sha(config_path))):
        if not metrics[key].eq(value).all():
            raise ValueError(f"Fold metrics disagree with the CV manifest: {key}.")
    spec = {"analysis_version": "v1", "source_run_id": report["manifest"]["run_id"],
            "source_cv_manifest": str(report["manifest_path"]), "source_cv_manifest_sha256": _sha(report["manifest_path"]),
            "source_tables_sha256": {name: _sha(report["run_dir"] / f"{name}.csv") for name in ("fold_metrics", "paired_accuracy")},
            "configuration_sha256": _sha(config_path), "policy_sha256": _sha(policy_path),
            "policy": policy, "primary_rope_half_width": rope,
            "source_sha256": {"snli_analysis.py": _sha(Path(__file__)), "analysis.py": _sha(Path(__file__).with_name("analysis.py")),
                              "baycomp/single.py": _sha(Path(baycomp.single.__file__))},
            "versions": {name: version(name) for name in ("baycomp", "scipy", "numpy", "pandas", "matplotlib")}}
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
    output = REPO_ROOT / config["paths"]["output_dir"]
    return config, policy, report, paired, spec, identity, output


def _load(directory, spec):
    manifest = _read(directory / "manifest.json")
    if manifest["status"] != "complete" or manifest["specification"] != spec:
        raise ValueError("Analysis is incomplete or uses different source results/code/policy. Run the analysis command.")
    for name, expected in manifest["artifact_sha256"].items():
        path = directory / name
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"Analysis artifact changed or is missing: {path}")
    return {"manifest": manifest, "manifest_path": directory / "manifest.json", "analysis_dir": directory,
            "tables": {name: pd.read_csv(directory / f"{name}.csv") for name in TABLES},
            "figures": {"overview": directory / "posterior_overview.png"},
            "posterior_figures": {row["contrast_id"]: directory / "posteriors" / f"{row['contrast_id']}.png"
                                  for row in manifest["contrasts"]}}


def run_bayesian_analysis(config_path=FROZEN_CONFIG, policy_path=ANALYSIS_POLICY):
    """Create immutable tables/plots, or check/reuse an identical completed run."""
    config, policy, cv, paired, spec, identity, output = _context(config_path, policy_path)
    root = output / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / identity
    contrasts = planned_contrasts(config)
    if not directory.exists():
        with tempfile.TemporaryDirectory(prefix=f".{identity}_", dir=root) as temporary:
            stage = Path(temporary)
            (stage / "posteriors").mkdir()
            rows, sensitivity, differences = [], [], []
            runs = len(config["seeds"]["cv_splits"])
            rope = config["evaluation"]["rope_half_width"]
            for contrast in contrasts.to_dict("records"):
                a, b = paired[contrast["system_a"]].to_numpy(), paired[contrast["system_b"]].to_numpy()
                result = contrast | correlated_posterior(a, b, runs=runs, rope=rope, credible_mass=policy["credible_mass"])
                rows.append(result)
                for width in sorted(set([rope, *policy["sensitivity_rope_half_widths"]])):
                    sensitivity.append(contrast | correlated_posterior(a, b, runs=runs, rope=width, credible_mass=policy["credible_mass"])
                                       | {"primary_rope": width == rope})
                difference = paired[KEYS].copy()
                difference["contrast_id"] = contrast["contrast_id"]
                difference["accuracy_a"], difference["accuracy_b"], difference["difference_b_minus_a"] = a, b, b - a
                differences.append(difference)
                _plot_posterior(result, stage / "posteriors" / f"{contrast['contrast_id']}.png")
            tables = {"paired_accuracy": paired, "contrasts": contrasts, "paired_differences": pd.concat(differences, ignore_index=True),
                      "posteriors": pd.DataFrame(rows), "rope_sensitivity": pd.DataFrame(sensitivity)}
            for name, table in tables.items():
                table.to_csv(stage / f"{name}.csv", index=False)
            _plot_overview(tables["posteriors"], stage / "posterior_overview.png")
            manifest = {"stage": "bayesian_paired_cv_analysis", "status": "complete", "analysis_id": identity,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(), "specification": spec,
                        "contrasts": contrasts.to_dict("records"), "primary_contrasts": int(contrasts["primary"].sum()),
                        "context_contrasts": int((~contrasts["primary"]).sum()),
                        "checks": {"source_complete_cv_checked": True, "identical_fold_order": True,
                                   "runs_equals_cv_repetitions": True, "original_rope_preserved": True, "test_not_scored": True},
                        "limitations": ["Correlated-normal paired-score model with the tutorial/baycomp matching-prior posterior.",
                                        "Approximate equal-fold correction: scale squared = sample variance * (1/N + 1/(K-1)).",
                                        "Single corpus, fixed group subset, hyperparameters and model seed; not initialization or cross-corpus uncertainty.",
                                        "Separate conditional pairwise probabilities, not a joint ranking or probability of being globally best."],
                        "artifact_sha256": {str(p.relative_to(stage)): _sha(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
            _write(stage / "manifest.json", manifest)
            # Only publish completed results; failed temporary work never replaces a report.
            stage.rename(directory)
    result = _load(directory, spec)
    pointer = output / "manifests/bayesian_analysis.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=pointer.parent, delete=False, encoding="utf-8") as stream:
        json.dump({"analysis_id": identity, "manifest_path": str(result["manifest_path"])}, stream, indent=2)
        temporary_pointer = Path(stream.name)
    temporary_pointer.replace(pointer)
    return result


def load_analysis_report(config_path=FROZEN_CONFIG, policy_path=ANALYSIS_POLICY):
    """Read/check completed analysis and its CV source without setup or fitting."""
    *_, spec, identity, output = _context(config_path, policy_path)
    pointer = output / "manifests/bayesian_analysis.json"
    if not pointer.is_file():
        raise FileNotFoundError("Run python -m part1.snli_analysis before inspecting section 8.")
    saved = _read(pointer)
    if saved["analysis_id"] != identity:
        raise ValueError("Analysis source/code/policy changed. Run python -m part1.snli_analysis.")
    directory = output / "analysis" / identity
    if Path(saved["manifest_path"]) != directory / "manifest.json":
        raise ValueError("Analysis pointer has an unexpected manifest path.")
    return _load(directory, spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--policy", type=Path, default=ANALYSIS_POLICY)
    args = parser.parse_args()
    report = run_bayesian_analysis(args.config, args.policy)
    print("Analysis:", report["manifest_path"])
    print(report["tables"]["posteriors"][["label", "mean_difference_pp", "p_a_better", "p_equivalent", "p_b_better"]].to_string(index=False))
