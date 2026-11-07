# Project 1: SNLI

Open [practical_1.ipynb](practical_1.ipynb). All custom code is written in its
cells; it does not import project helper modules or read external experiment
configurations/results. The notebook has one fixed-setting experiment:

**Training + validation → shared 2k BPE → grouped CV → best model → full refit → test.**

The nine candidates combine indicators, unigram counts or pure bigram counts
with logistic regression or a one-/two-hidden-layer MLP. Settings are in one
`BASE` dictionary. There is no hyperparameter search or early stopping. Final
neural training uses the same three epochs as CV.

## Run

Select `.venv/bin/python` as the notebook kernel, or launch Jupyter:

```bash
source .venv/bin/activate
jupyter lab part1/practical_1.ipynb
```

**Run All performs fresh fitting**, including 45 CV fits and the selected final
model. It saves results in notebook outputs only, with tables and graphs inline.
Save the notebook after execution. No trial logs, checkpoints or result tables are written to files.
Plots are also exported as PNG and PDF to `figures/project1_2k/`. Existing cleaned SNLI data are reused; a fresh checkout downloads the
pinned revision through Hugging Face. Data remain excluded from Git.

For a new environment, use Python 3.12 and install the root `requirements.txt`
(or the recorded `environment.txt` snapshot with the CPU PyTorch index described
there). The repaired current environment uses the `.venv` interpreter.

## Read the results

| Section | Content |
| --- | --- |
| 1 | One frozen configuration |
| 2 | Source splits, eligible development pool and grouped CV subset |
| 3–4 | Shared BPE, sparse representations, models and fixed training loop |
| 5 | All nine CV results, comparison graph and automatic winner selection |
| 6 | Paired Bayesian comparisons and visible ROPE sensitivity table |
| 7 | Selected model's out-of-fold errors |
| 8 | Full-pool refit, official-test table, per-class scores and graphs |
| 9 | Tutorial-style fold boxplots/points, Bayesian posterior curves and ROPE heatmap |

BPE is trained once on **559,164 eligible training + validation pairs**, excluding
test-premise overlaps. CV uses **60,002 rows**, one five-fold split and model seed
0. Because BPE sees CV held-out text without labels, these scores are conditional
on the shared tokenizer. Bigram inventories and model weights use fold-training
rows only. The official test is excluded from learned preprocessing.

Earlier work already evaluated another model on test. The notebook states that
history explicitly; it does not substitute historical test scores for this fit.
Historical Python modules, JSON configurations and `practical_1_details.ipynb`
remain reference material and are not dependencies of the current experiment.

## Completed fresh results

The selected **indicator MLP2** achieved **62.60% mean CV accuracy** and
**74.48% test accuracy**, with test NLL **0.6210** and macro-F1 **0.7439**.
CV took **3.09 minutes**; full preparation/refitting took **51.54 seconds**.
The notebook contains all nine candidates and their uncertainty analysis.

Seven figure sets are displayed inline and exported in
[figures/project1_2k/](figures/project1_2k/) as PNG and PDF. In particular:

- [Fold boxplots with individual scores](figures/project1_2k/cv_boxplots_and_points.png)
- [Bayesian posteriors and ROPE](figures/project1_2k/bayesian_posteriors_rope.png)
- [ROPE sensitivity](figures/project1_2k/rope_sensitivity.png)
- [Test confusion matrices](figures/project1_2k/test_confusion.png)
- [Training curves](figures/project1_2k/training_curves.png)

## Verification and sources

```bash
.venv/bin/python -m unittest discover -s part1/tests -v
```

The notebook tests cover literal inline definitions, test exclusion, complete
premise groups, feature values/boundaries, fixed epochs, shared-tokenizer CV,
selection and Bayesian direction. Other tests cover the retained historical modules.
See [requirements](project.ipynb), [plan](PROJECT1_PLAN.md),
[findings](PROJECT1_FINDINGS.md), [course methods](COURSE_METHODS.md) and
[AI assistance](AI_ASSISTANCE.md).
