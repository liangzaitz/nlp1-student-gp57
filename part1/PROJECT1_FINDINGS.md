# Project 1 findings: shared 2k BPE

Updated 2026-11-07. These are the **fresh completed results** in
[practical_1.ipynb](practical_1.ipynb), using one fixed-setting experiment.
Historical pilot/tuning scores and previous test results are not part of these tables.

## Data and fixed design

SNLI revision `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b` is pinned.
Labels are `[entailment, neutral, contradiction]`, IDs `[0, 1, 2]`.

| Pool | Rows | Use |
| --- | ---: | --- |
| Clean official train | 549,367 | Source data |
| Clean official validation | 9,842 | Source data |
| Eligible train + validation | 559,164 | Shared BPE and final classifier fitting |
| Fixed CV subset | 60,002 | Complete premise groups, target 60,000, sampling seed 0 |
| Clean official test | 9,824 | Final selected-system evaluation |

Exact premises shared with test are excluded from the development pool. Grouped
folds keep all pairs sharing an exact premise together; this does not remove all
possible semantic near-duplicates. Test-premise membership is used only for
exclusions; no test text enters learned preprocessing.

One BPE is trained on all eligible development text, from both sentence roles:
**2,000 actual tokens**, minimum frequency 2, Metaspace, no case normalization.
It is fixed throughout CV and final refitting. Each fold learns a separate
training-only bigram inventory, frequency ≥2 and at most **2,000 types**.
Inputs are `[premise features; hypothesis features]`, with up to 4,000 columns.
Pure bigrams contain no unigram fallback.

CV uses **five StratifiedGroupKFold folds, split seed 0**, with the same partitions
for all nine candidates. Model seed is 0. Logistic uses L2/SAGA, C=1, maximum
1,000 iterations and tolerance 0.0001. MLPs use `[128]` or `[128,64]` ReLU units,
dropout 0.1, AdamW learning rate 0.001, weight decay 0.0001 and batch size 128.
Neural training uses **three fixed epochs in both CV and the final fit**.
There is no hyperparameter search, early stopping or accuracy-based checkpoint selection.

The tokenizer sees CV held-out text without labels, so CV scores are conditional
on shared development-fitted BPE, rather than an entirely fold-fitted pipeline.
This design was explicitly requested. No claim is made that these settings are optimal.

## Fresh CV results

Equal-weight means over five folds; SD is descriptive fold variation in percentage
points, not a confidence interval or independent-sample standard error.
All 15 logistic fits met the configured convergence criterion. There were **45 fits**.
CV took **3.09 minutes**; shared BPE fitting took **2.57 seconds**.

| Features | Classifier | Mean accuracy | Fold SD (pp) | NLL | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Indicators | MLP2 | **62.60%** | 0.63 | **0.8394** | 0.6245 |
| Unigram counts | MLP2 | 62.59% | 0.79 | 0.8408 | 0.6243 |
| Unigram counts | MLP1 | 61.95% | 0.68 | 0.8459 | 0.6183 |
| Indicators | MLP1 | 61.92% | 0.49 | 0.8457 | 0.6179 |
| Indicators | Logistic | 60.52% | 0.51 | 0.8965 | 0.6043 |
| Unigram counts | Logistic | 60.41% | 0.52 | 0.8965 | 0.6032 |
| Pure bigram counts | MLP1 | 53.78% | 0.25 | 0.9499 | 0.5367 |
| Pure bigram counts | MLP2 | 53.66% | 0.33 | 0.9533 | 0.5338 |
| Pure bigram counts | Logistic | 52.75% | 0.31 | 0.9778 | 0.5265 |

The stated rule automatically selects **indicators + the two-hidden-layer MLP**:
highest unrounded mean accuracy, with NLL breaking an exact tie. It also has the
lowest mean NLL here. The near tie with unigram MLP2 does not establish a useful
practical advantage. Pure bigrams underperform under this particular fixed cap
and schedule; the result does not show that all bigram approaches are inferior.

## Bayesian analysis and ROPE

Use five paired scores and the Bayesian correlated-t posterior, with differences
**B − A**, `df=4`, and Student scale squared
`sample_variance(differences) * (1/5 + 1/4)`. The implementation agrees with the
underlying `baycomp` library used by the tutorial. There are **18 one-factor
comparisons** and three ROPE widths, **54 sensitivity rows**.

The primary ROPE is **±0.5 percentage points**: a project judgement about the
smallest practically interesting accuracy difference. Sensitivity widths are
±0.25 and ±1 pp. Pairwise probabilities do not give global ranking probabilities.

- Indicator MLP2 versus unigram MLP2: **97.75%** probability inside the primary ROPE.
- Indicators versus unigram counts with MLP1: **97.95%** inside the primary ROPE.
- Unigram MLP2 minus MLP1: mean **+0.64 pp**, 95% interval **+0.02 to +1.25 pp**;
  P(B better beyond the ROPE) **71.51%**, P(ROPE) **28.15%**.
- For that depth comparison, equivalence probability changes from **6.95%** to
  **28.15%** to **91.14%** as the half-width increases from 0.25 to 0.5 to 1 pp.
- Pure bigram MLP1 versus unigram MLP1: mean **−8.17 pp**, with a 95% interval
  **−9.72 to −6.62 pp**. This difference persists across the chosen ROPE widths.

Only one CV repetition is used. The correlated-normal/common-correlation model
is an approximation, and five dependent scores give limited information. These
probabilities describe this subset, shared tokenizer, fixed settings and splits.
They do not override the predeclared accuracy-based selection rule.

## Full refit and official test

Fit a fresh indicator MLP2 on all **559,164** eligible development pairs, using the
same 2k BPE, model seed 0 and three epochs. Input dimension is **4,000**, with
hidden layers **128 → 64**. Final preparation/fitting took **51.54 seconds**.
Then generate one test probability set for this fitted model; all test plots and
reports reuse those predictions.

| Test metric | Result |
| --- | ---: |
| Accuracy | **74.48%** (7,317 / 9,824 correct) |
| NLL | **0.6210** |
| Macro-F1 | **0.7439** |

| Gold class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Entailment | 0.7492 | 0.8168 | 0.7815 | 3,368 |
| Neutral | 0.6920 | 0.7015 | 0.6967 | 3,219 |
| Contradiction | 0.7989 | 0.7130 | 0.7535 | 3,237 |

Confusion matrix, rows gold and columns predicted in the class order above:

| | Entailment | Neutral | Contradiction |
| --- | ---: | ---: | ---: |
| Entailment | 2,751 | 444 | 173 |
| Neutral | 553 | 2,258 | 408 |
| Contradiction | 368 | 561 | 2,308 |

Training curves are online minibatch measurements, with dropout active and
parameters changing throughout each epoch. Their last accuracy is about **76.4%**;
it is not held-out accuracy or the final model evaluated on the complete training
pool. CV classifiers saw about 48,000 labelled pairs per fold, whereas the final
classifier saw 559,164; the evaluation examples also differ. The CV-to-test change
is not an isolated estimate of the benefit of more training data.

## Error review and limitations

The notebook displays the first 12 OOF errors for the selected system and the
first 12 test errors, in fixed source order. These are illustrative cases, not a
random sample for estimating linguistic error-category frequencies.

- A choir singing at church versus a choir singing at a baseball game is gold
  contradiction but predicted entailment: the model fails a venue conflict despite
  shared activity words.
- A woman with a green headscarf versus “The woman is young” is gold neutral but
  predicted entailment: age is unsupported by the premise.
- A Land Rover crossing a river versus splashing water while crossing is gold
  entailment but predicted neutral: the inference needs more than matching words.

These examples illustrate limitations in sentence interaction and contextual
inference; they do not establish the causal mechanism of every error. Students
should review the examples, interpretations and gold-label conventions.

Earlier experiments already explored tuning and evaluated a different model on
test. The interrupted shared-8k run completed no new test evaluation. This fresh
2k experiment follows the newly agreed fixed protocol; it must not be described
as a never-before-seen first test. Because several design elements differ from
earlier experiments, it is not a controlled vocabulary-size ablation.

## Plots and verification

All seven figure sets are inline and exported as PNG and PDF under
[figures/project1_2k](figures/project1_2k/): CV means, tutorial-style fold boxplots
with points, Bayesian posterior/ROPE curves, ROPE sensitivity heatmap, test
confusions, per-class metrics and training curves.

The main notebook has **31 cells, 18 code cells and 710 code lines**. Every code cell executed in a fresh kernel with
no error, warning or widget outputs. No project helper modules, external result
files or JSON configurations are loaded. The notebook writes figures only;
experiment tables stay in its outputs.

The 62-test repository suite passed, followed by an additional passing check of
the new posterior scale/probabilities against the tutorial library (**63 checks**
in total). Package consistency also passed. These checks establish software
behaviour, not student review or scientific optimality. Group code/error review
and oral preparation remain. See [AI assistance](AI_ASSISTANCE.md).
