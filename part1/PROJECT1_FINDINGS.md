# Project 1 findings: fold-trained BPE and vocabulary size

Updated 2026-11-07. These are the completed results in
[practical_1.ipynb](practical_1.ipynb). **Only BPE vocabulary size was tuned**:
1,000 / 2,000 / 3,000. The final test score did not determine the selection.

## Design and fitting boundaries

SNLI revision `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b` is pinned. The cleaned
source splits have 549,367 training, 9,842 validation and 9,824 test pairs.
Combining training + validation and excluding 45 exact-test-premise overlaps gives
**559,164 eligible development pairs**. Exact-premise grouping controls repeated
premises, not all possible semantic near-duplicates.

CV uses the unchanged **60,002-row** complete-group sample, sampling seed 0, and
five StratifiedGroupKFold partitions, split seed 0. Each classifier trains on
about 48,000 pairs and validates on about 12,000. All candidates use the same folds.

For each fold/size, fit **BPE on that training partition's text only**, then its
bigram inventory. Encode held-out text using those fixed mappings. Share the
fitted preprocessing across its nine representation/classifier combinations.
There are **15 CV tokenizers and 135 classifier fits**. Different folds may use
different coordinates; each individual pipeline remains internally consistent.

Other settings remain fixed: minimum BPE merge frequency 2, `[UNK]`, Metaspace,
preserved case/punctuation, bigram minimum frequency 2 and **cap 2,000 at all
sizes**. Inputs concatenate premise and hypothesis feature blocks. Pure bigrams
have no unigram fallback. Logistic uses L2/SAGA, C=1, max_iter=1000, tolerance
0.0001. Neural models use `[128]` or `[128,64]`, ReLU/dropout 0.1, AdamW learning
rate 0.001/weight decay 0.0001, batch 128 and **three fixed epochs**, model seed 0,
CPU/one PyTorch thread. There is no early stopping or other hyperparameter search.

## Best tested system per vocabulary

These are equal-weight mean CV scores over five folds. Per-size winners may use
different representations; differences between these rows are not vocabulary-only
contrasts. The next table holds each representation/classifier fixed instead.

| BPE size | Best system | Mean accuracy | Fold SD (pp) | NLL | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1,000 | Unigram counts / MLP2 | 62.27% | 0.70 | 0.8415 | 0.6207 |
| **2,000** | **Indicators / MLP2** | **62.69%** | **0.65** | **0.8400** | **0.6253** |
| 3,000 | Indicators / MLP2 | 62.55% | 0.76 | 0.8469 | 0.6237 |

The fixed rule selects **2,000 BPE tokens + indicators + MLP2**, using highest
unrounded mean accuracy and lower NLL only for an exact accuracy tie. Further
exact ties use smaller vocabulary, then lexical representation/classifier order.
This is the best tested candidate under the stated settings, not a proven optimum.

## Controlled vocabulary comparison

Mean CV accuracy for the same representation/classifier at each size:

| Features | Classifier | 1,000 | 2,000 | 3,000 |
| --- | --- | ---: | ---: | ---: |
| Indicators | Logistic | 59.77% | 60.47% | 60.52% |
| Indicators | MLP1 | 61.32% | 61.93% | 62.18% |
| Indicators | MLP2 | 62.25% | **62.69%** | 62.55% |
| Unigram counts | Logistic | 59.74% | 60.53% | 60.43% |
| Unigram counts | MLP1 | 61.44% | 62.09% | 62.28% |
| Unigram counts | MLP2 | 62.27% | 62.54% | 62.50% |
| Pure bigram counts | Logistic | 54.82% | 52.95% | 52.29% |
| Pure bigram counts | MLP1 | 55.82% | 53.91% | 53.25% |
| Pure bigram counts | MLP2 | 55.80% | 54.03% | 53.35% |

Larger vocabularies are not uniformly better. Indicator/unigram MLP1 improves
through 3k, whereas both MLP2 versions peak at 2k. Every pure-bigram classifier
performs best at 1k under the fixed 2k inventory cap. This concerns these feature
caps and training settings, not the general usefulness of bigrams.

The notebook includes all 27 summaries with NLL/F1, deltas relative to 2k,
dimensions, parameter counts, mean fit time and a collapsed table of all 135 fold
measurements. Fold SD describes dependent-fold variability, not a confidence interval.

## Segmentation and model cost

All folds reached their requested vocabulary sizes and retained 2,000 bigram types.
Validation token lengths count both sentence roles together. Costs below keep
**indicators / MLP2** fixed; fit seconds exclude preprocessing and prediction and
are descriptive CPU observations, not a controlled performance benchmark.

| BPE size | Mean validation tokens/pair | Input dimensions | Parameters | Mean classifier fit (s) |
| --- | ---: | ---: | ---: | ---: |
| 1,000 | 31.00 | 2,000 | 264,579 | 2.33 |
| 2,000 | 26.57 | 4,000 | 520,579 | 3.93 |
| 3,000 | 24.77 | 6,000 | 776,579 | 5.67 |

Bigger BPE shortens token sequences while enlarging indicator/unigram feature
vectors and models. Pure bigram input remains 4,000 dimensions at every size.
Thus the experiment tunes vocabulary size as a pipeline setting, including its
segmentation and model-size effects. It does not hold neural parameter count fixed.

## Paired Bayesian evidence

The tutorial's correlated-t posterior uses five matched fold differences **B−A**,
`df=4` and scale squared `sample_variance * (1/5 + 1/4)`. The primary ROPE is
±0.5 percentage points, with ±0.25 and ±1 pp sensitivity. This is a project
judgement about practical relevance. Comparisons are descriptive **post-selection**
evidence using the same CV scores, not independent confirmation or global ranks.

For the fixed selected system, indicators / MLP2:

| A → B | Mean difference (pp) | 95% interval (pp) | P(A better beyond ROPE) | P(inside ROPE) | P(B better beyond ROPE) |
| --- | ---: | --- | ---: | ---: | ---: |
| 1k → 2k | +0.43 | [−0.31, +1.18] | 1.28% | 57.89% | 40.83% |
| 1k → 3k | +0.29 | [−0.30, +0.89] | 1.03% | 79.59% | 19.39% |
| 2k → 3k | −0.14 | [−0.94, +0.66] | 13.92% | 81.58% | 4.49% |

These do not establish strong practical superiority of 2k. Its selection follows
the fixed highest-accuracy rule. For 2k versus 3k, equivalence probability changes
from **51.63% / 81.58% / 97.16%** at half-widths 0.25 / 0.5 / 1 pp.
There are 18 one-factor system comparisons at the selected size, three vocabulary
comparisons for the selected system, and **63 sensitivity rows** in total.

The correlated-normal/common-correlation model is an approximation. One five-fold
repetition provides limited information. Selecting among 27 candidates makes the
maximum CV mean optimistic as an estimate of the complete selection procedure;
there is no nested outer CV in this budgeted experiment.

## Final refit and official test

After selecting from CV, fit a **new 2k BPE** and fresh indicator MLP2 on all
559,164 development pairs. Keep the same three epochs and other settings.
CV took **10.09 minutes**; final preprocessing/fitting took **60.25 seconds**.
All **45 logistic fits** met the configured convergence criterion.

Only the selected final pipeline was evaluated on official test in this run.
It selects the same final recipe as the earlier shared-2k experiment, and its
fresh refit reproduces the same test result. This is not a test-accuracy gain from
tuning. The CV preprocessing boundary has changed, while final fitting in both
experiments learns preprocessing on all eligible development text.

| Test metric | Result |
| --- | ---: |
| Accuracy | **74.48%** (7,317 / 9,824) |
| NLL | **0.6210** |
| Macro-F1 | **0.7439** |

| Gold class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Entailment | 0.7492 | 0.8168 | 0.7815 | 3,368 |
| Neutral | 0.6920 | 0.7015 | 0.6967 | 3,219 |
| Contradiction | 0.7989 | 0.7130 | 0.7535 | 3,237 |

The test confusion matrix is rows gold/columns predicted in that class order:
`[[2751,444,173],[553,2258,408],[368,561,2308]]`.
Online training accuracy reaches about 76.4%; it is measured during updates with
dropout active, not on held-out data. CV fits train on about 48k labels versus
559k at final refitting, and evaluate different examples; the CV-to-test difference
is not an isolated effect of training size.

The notebook shows 12 selected-system OOF errors and 12 test errors in fixed
source order. Examples include a choir's church/baseball venue conflict predicted
as entailment and an unsupported age claim (“The woman is young”) predicted as
entailment. These illustrate contextual-inference failures; they do not establish
one cause for all errors or category prevalence. Student review remains required.

Earlier project work already inspected test. This fresh run keeps test scores out
of its selection, but is not an untouched first test evaluation. Further setting
changes should use development evidence.

## Plots and verification

Nine figure sets are inline and exported as PNG/PDF under
[figures/project1_vocab_cv/](figures/project1_vocab_cv/), including vocabulary
accuracy/NLL curves, selected-size fold boxplots/points, system and vocabulary
posterior/ROPE curves, sensitivity heatmap, test confusions/class metrics and
training curves. The historical shared-2k figure folder is not used here.

All 64 repository tests passed. The main notebook executed in a fresh kernel
without error/warning/widget outputs. An independent audit recomputed selection
from the 135 embedded fold records and checked pairing, dimensions, parameter
counts and integer correct-count consistency. Each fold/size's BPE training-row
membership is checked by the notebook tests. All implementation remains literal
in notebook cells, with no project helper imports or external experiment-result
reads. Only figures are written; tables remain in notebook outputs.

Group code/error review and oral preparation remain. See [plan](PROJECT1_PLAN.md)
and [AI assistance](AI_ASSISTANCE.md).
