# Project 1: SNLI

Start with [practical_1.ipynb](practical_1.ipynb). It follows the supplied
[course practical](https://github.com/S-Chaudhuri/uva_nlp1_2025/blob/main/NLP1_2025_Practical_2_%28student%29.ipynb):
short explanations, worked inputs, explicit model/training code, plots and interpretation.
All custom code used by this notebook is written in its cells. It does not import
local project modules, execute Python files, or load another notebook's definitions.
Installed packages, JSON configurations, downloaded data and saved experiments
remain dependencies; the notebook is not a standalone data/result distribution.

## Open and run

From the repository root, use the existing environment:

```bash
source .venv/bin/activate
jupyter lab part1/practical_1.ipynb
```

In VS Code select `.venv/bin/python` as the notebook kernel, then **Run All**.
The notebook finds the repository from either its root or `part1`.
It reads the completed experiments, verifies their results, and runs 30 teaching
updates on 16 training examples. **It starts no new training or test-inference pass.** Final test probabilities
are already saved; Run All checks them and regenerates their graphs. Cached Arrow files contain the downloaded SNLI data;
offline loading and cache lock files are normal. Widget progress bars are disabled.

The current checkout contains the ignored data/results. A new checkout needs the
data and completed artifact directories to reproduce these displays. The detailed
notebook and earlier CLI modules preserve the original experiment-generation
workflow; commands for that workflow are in the locally preserved previous README
under `results/project1/review/before_streamlining/`. Generating all baseline
artifacts is separate from the default read-and-check Run All.

The verified environment is Python 3.12.3 on Linux with CPU PyTorch; the course
README specifies Python 3.14. To recreate our recorded environment on a compatible
machine, create a Python 3.12 virtual environment and install the snapshot:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r part1/environment.txt
python -m pip check
```

## Reading order and current status

| Notebook section | Content |
| --- | --- |
| 1–3 | Setup, data, leakage-aware pools and grouped folds |
| 4 | BPE, indicators, unigram counts and pure bigram counts |
| 5 | Models, one training loop, gradient example, bundles and logits → probabilities → decision |
| 6 | Short joint tokenizer/model search and checked validation results |
| 7–8 | Checked baseline CV scores and Bayesian/ROPE comparisons |
| 9 | One-repetition OOF error review, 50 annotated cases and frozen final recipe |
| 10 | Completed final fit/test, learning curves and test graphs |
| 11 | Group review and oral preparation |

The baseline has **165 completed fits** (135 primary, 30 diagnostic/baseline).
The short search has **27 completed observations**, nine reused and 18 newly
fitted, taking **7m 58s** for that invocation. CV and validation use different
training sizes and evaluation sets; their accuracies are not a direct before/after
comparison. Step 9 now adds 50 AI-assisted error notes, class confusions, text
slices and shared-error counts. The final 5k unigram MLP1 recipe is frozen at five
epochs in [snli_final.json](configs/snli_final.json). The final fit completed in
2.98 minutes and achieved **74.58% test accuracy**, macro-F1
**0.7451**, NLL **0.6886** on 9,824 cleaned test rows. See [findings](PROJECT1_FINDINGS.md) and the
[remaining checklist](PROJECT1_PLAN.md).

## Optional new experiments

Keep `RUN_SEARCH = False`, `RUN_CV = False` and `RUN_FINAL = False` for the normal report. To run a new
experiment, save the notebook and enable only its corresponding switch.
New outputs use `results/project1/streamlined/`, preserving historical artifacts.

`run_search` tests the three saved joint recipes on the 100,001-row training
sample and all 9,839 eligible validation rows. It fits training-only preprocessing,
uses one seed, at most eight neural epochs and patience three, restoring the best
accuracy checkpoint with NLL as a tie-break. Its simple resume checks its own
completed trials. It does not import historical trials or run the extended policy.
An explicit resume gets another allowance; none is scheduled automatically.

Both runners cap each invocation at 900 seconds, including preprocessing. Python
signal delivery can lag in native library calls. The completed baseline originally
took 21.62 minutes, so a fresh capped CV invocation may be incomplete. Only complete
paired scores support the reported comparisons; the simplified CV runner requires
a new directory rather than resuming partial grids. Historical robust CV resumption,
trial-import checks, pilot and optional extensive search live in
[practical_1_details.ipynb](practical_1_details.ipynb), which the main notebook never
imports. Its outputs are cleared and training switches are false.

The completed final experiment is separate in `results/project1/final/snli_5k_unigram_mlp1/`.
Its source/configuration snapshot, exact training rows, complete model and test
probabilities are retained. Even an explicit `RUN_FINAL` invocation reuses a
completed matching run; changed/corrupt artifacts are rejected. Four PNG graph
sets live in its `figures/` directory and appear in notebook section 10.

## Verification and supporting records

```bash
.venv/bin/python -m unittest discover -s part1/tests -v
```

The streamlined notebook was executed in a fresh kernel with project-helper
imports blocked. Its checks recompute metrics from all 165 CV prediction sets
and all 27 validation trials, replay 11 saved systems from raw text, and recompute
28 Bayesian contrasts plus 84 ROPE-sensitivity rows. Synthetic tests cover feature
boundaries, both training modes, bundle reloads, search resume, a complete small CV
and the timer, plus error-analysis alignment/slicing/sampling. Software-test scores are not SNLI experiment results.

- [Project requirements](project.ipynb) and [tutorial](tutorial.ipynb).
- [Course-method map](COURSE_METHODS.md), including lecture page references.
- [Configurations](configs/README.md) and [environment snapshot](environment.txt).
- [AI-assistance record](AI_ASSISTANCE.md); students still need to review and explain the work.
