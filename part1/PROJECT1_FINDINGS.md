# Project 1 findings: current checked report

Updated 2026-11-07. The working report is [practical_1.ipynb](practical_1.ipynb).
Baseline CV/Bayesian analysis and the short validation search are complete.
**Steps 1–10 are complete, including the frozen final fit and one-time official
test evaluation.** Group review and oral preparation remain. No additional
hyperparameter search was run for the final fit.

## Data and comparison design

SNLI revision `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b` is pinned.
Labels are `[entailment, neutral, contradiction]`, IDs `[0, 1, 2]`.

| Pool | Rows | Use |
| --- | ---: | --- |
| Clean official train | 549,367 | Source training data |
| Clean official validation | 9,842 | Source development data |
| Clean official test | 9,824 | Completed frozen-model final evaluation |
| Eligible train | 549,253 | Exclude premises in either holdout |
| Eligible validation | 9,839 | Exclude premises in test |
| Baseline CV sample | 60,005 | Whole-premise-group sample, seed 0 |
| Short-search training sample | 100,001 | Whole-premise-group sample, seed 57 |
| Final refit pool | 559,164 | Train + validation excluding test premises |

Cleaning removed 1,119 invalid-label rows: 785 / 158 / 176 by source split.
No additional blank-text rows were removed. Exact-premise groups remain together
in CV; this controls repeated premises, not every possible semantic near-duplicate.
Before the final fit, only test structure/premises were used for eligibility;
test performance was first evaluated after recipe/schedule freeze in Step 10.

The baseline uses five grouped folds × three split seeds `[0,3,5]`, model seed 0,
and shared training-only BPE/inventory within each fold. BPE target vocabulary and
bigram cap are both 5,000, merge/bigram minimum frequency 2, with preserved case.
Inputs are separate premise/hypothesis blocks: binary indicators, unigram counts,
or **pure** retained-bigram counts. Bigram input has no unigram fallback and its
units may be subwords. At `V=B=5000`, all inputs have 10,000 coordinates.

Logistic uses SAGA, C=1 and at most 1,000 iterations. MLPs use 128 or 128→64
ReLU units, dropout .1, AdamW lr .001/weight decay .0001, batch 128, CPU/one thread.
The earlier small engineering pilot motivated a fixed **three-epoch** schedule.
Outer held-out scores never select baseline checkpoints. It was not thorough
hyperparameter optimization.

## Baseline grouped CV

Run `713ab1f5b3cb5801`, protocol `3e3b8a39db826f1c`: 135 primary fits plus
15 class-prior and 15 hypothesis-only fits. Recorded execution took 21.62 minutes;
all 60 logistic fits converged under the configured criterion. Values below are
**equal-weight means over 15 folds**. Fold SD is descriptive, not a confidence
interval; overlapping training sets and repeated examples create dependence.

| Features | Model/input | Accuracy | Fold SD (pp) | NLL | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Indicators | MLP2 | 62.75% | 0.451 | 0.8514 | 0.6261 |
| Unigrams | MLP2 | 62.72% | 0.330 | 0.8503 | 0.6257 |
| Unigrams | MLP1 | 62.45% | 0.386 | 0.8466 | 0.6235 |
| Indicators | MLP1 | 62.35% | 0.362 | 0.8470 | 0.6225 |
| Unigrams | Logistic | 60.76% | 0.266 | 0.9213 | 0.6069 |
| Indicators | Logistic | 60.75% | 0.240 | 0.9199 | 0.6069 |
| Pure bigrams | MLP1 | 55.32% | 0.410 | 0.9311 | 0.5522 |
| Pure bigrams | MLP2 | 55.08% | 0.440 | 0.9401 | 0.5486 |
| Pure bigrams | Logistic | 53.75% | 0.408 | 0.9833 | 0.5370 |
| Unigrams | Logistic (H only) | 61.02% | 0.334 | 0.8890 | 0.6095 |
| Unigrams | Class prior | 33.37% | 0.005 | 1.0986 | 0.1668 |

The highest mean accuracy is 62.75% (indicator MLP2); the lowest mean NLL is
0.8466 (unigram MLP1). Indicators and raw counts are close. Pure bigrams perform
worse with this fixed inventory, plausibly reflecting sparse coverage and the
loss of standalone token evidence; this is not a claim that all bigram methods
are inferior.

The hypothesis-only diagnostic reaches 61.02%, compared with 60.76% for pair-input
unigram logistic. It uses the shared pair-trained tokenizer and only the hypothesis
feature block. This motivates checking label cues and the classifier's limited
interactions. It does not prove that premises are unnecessary or establish the
cause/prevalence of annotation artifacts. Being above the 33.37% prior baseline
alone does not establish strong semantic reasoning.

## Bayesian practical comparisons

Analysis `3b4f7bdb5c7c300b` uses all 15 matched accuracy scores for each system,
with `runs=3`, `k=5`, differences **B−A** and `df=14`.
For fold differences `d`, Student scale² is `var(d, ddof=1) × (1/15 + 1/4)`.
That is 4.75 times the naive independent-score scale². Posterior SD additionally
multiplies Student scale by `sqrt(14/12)`.

The primary ROPE was preselected as **±0.005 accuracy = ±0.5 percentage points**:
a project judgement about practically useful gains, not a sampling-error margin.
There are 18 primary one-factor contrasts and 10 separate context contrasts.
Sensitivity at ±0.25/±0.5/±1 pp preserves the original primary threshold.

- Indicators and unigram counts usually have high probability inside the ROPE.
- Pure bigrams lose roughly 7–8 pp against the corresponding unigram/indicator recipes.
- MLPs improve accuracy over logistic; extra depth has less decisive practical value.
- Unigram MLP2−MLP1: +0.266 pp, **95.97%** probability inside the ROPE.
- Hypothesis-only−pair unigram logistic: +0.258 pp, **97.72%** probability inside it.

Sensitivity is visible in notebook section 8 as a three-column equivalence table.
For unigram MLP1→MLP2, P(inside ROPE) is 44.89% / 95.97% / 100.00% (rounded)
at ±0.25 / ±0.5 / ±1 pp. The close-depth conclusion depends on our practical
tolerance; the underlying posterior is unchanged. Large pure-bigram losses persist
across the thresholds. The primary ROPE stays ±0.5 pp.

These are conditional pairwise posterior statements, not global ranking
probabilities. The correlated-normal/common-correlation model is an approximation;
fold sizes vary slightly and accuracy is bounded. Split seeds are not independent
neural-initialization replications. All conclusions concern this subset and recipes.

## Short joint tokenizer/model search

Completed run `4baddae6266ac867`: 27 observations, **nine explicitly reused and
18 new fits**, in **478.43 seconds (7m 58s)** for that invocation. The reuse
lineage is retained with original hashes. This runtime excludes the earlier work
that produced the reused trials; it is not a fresh 27-fit runtime estimate.
The stopped large run and optional extended policy are historical, not queued work.

Each system tests three joint recipes: baseline 5k BPE, compact/lowercase 3k with
stronger regularization, and larger 8k with merge minimum 5/faster learning.
Other settings include bigram inventory, C, widths, dropout, decay and batch size.
All use the same 100,001 training / 9,839 validation rows and model seed 0.
Neural selection uses at most eight epochs, patience three, best validation
accuracy and NLL as an exact-tie break. Nonconverged logistic trials are ineligible.

| Features | Model | Best tested recipe | Validation accuracy | NLL | Best epoch | Ceiling review |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Indicators | Logistic | baseline | 63.53% | 0.8443 | — | no |
| Indicators | MLP1 | baseline | 67.50% | 0.8351 | 5 | no |
| Indicators | MLP2 | baseline | 67.66% | 0.8671 | 4 | no |
| Unigrams | Logistic | baseline | 63.52% | 0.8549 | — | no |
| Unigrams | MLP1 | larger_faster_learning | 67.83% | 0.8624 | 4 | no |
| Unigrams | MLP2 | compact_regularized | 67.32% | 0.8775 | 7 | yes |
| Pure bigrams | Logistic | compact_regularized | 59.93% | 0.8709 | — | no |
| Pure bigrams | MLP1 | compact_regularized | 60.69% | 0.8637 | 2 | no |
| Pure bigrams | MLP2 | compact_regularized | 61.10% | 0.9918 | 5 | no |

The maximum validation accuracy is **67.83%**, from the larger-recipe unigram
MLP1. Its same-pool baseline-hyperparameter counterpart achieves **67.74%**;
the selected gain is only **0.09 pp**, with worse NLL (0.8624 versus 0.8344).
This does not establish a robust generalization gain. Its validation NLL rises
while training NLL falls; the accuracy-selected epoch 4 does not minimize NLL.

Compact pure-bigram recipes improve more: 59.93/60.69/61.10% for logistic/MLP1/MLP2,
versus 56.01/58.69/58.61% for the same-pool baseline recipes. Several settings
change together, so these differences do not isolate a tokenizer effect.

The selected compact unigram MLP2 has an active ceiling-review flag: its best
checkpoint is epoch 7 of 8, without exhausting patience. The budget-limited
search has one seed, no full-data confirmation and only three joint recipes per
system. It finds best tested validation recipes, not proven optima.

**Do not compare 67.83% validation with 62.75% CV as a tuning gain.** Training
sizes and evaluation protocols differ. No new tuned CV/Bayesian scores exist.
The original CV comparison remains the main assignment evidence; a new tuned-CV
claim needs an appropriate selection/evaluation design inside outer training folds.

## Report review and simplification

The old report mixed current evidence with stale statements that tuning was
pending, module-import workflows and successive implementation histories.
Current claims distinguish completed baseline CV/validation selection/error review,
optional tuned CV, and the completed frozen-model final test.

| Notebook | Cells | Code cells | Code lines |
| --- | ---: | ---: | ---: |
| Previous main notebook | 219 | 143 | 3,211 |
| Streamlined main notebook (through Step 10) | 70 | 42 | 1,297 |
| Supplied course reference | 145 | 76 | 954 |

The reference's short explanations, worked examples and visible training loops
informed presentation; its different task and sequence architectures were not
adopted. The main report now has one encoder, one training function, one simple
search, one CV loop and direct correlated-t calculation. All custom code it uses
is literal cell code: no local implementation imports, dynamic file execution or
loading another notebook. Data/configurations/saved artifacts remain dependencies.
The original experiment implementation is preserved in
[practical_1_details.ipynb](practical_1_details.ipynb), with outputs cleared.
Earlier long-form notes and the full pre-review notebook are preserved locally
under ignored `results/project1/review/` and `results/project1/notebooks/`.

Verification on 2026-11-06:

- **69 repository tests passed**, including 14 tests of the literal notebook definitions.
- **All 42 main code cells executed in a fresh kernel**, with project-helper imports
  blocked, no cell errors and no widget-output dependencies.
- Independently recomputed accuracy/NLL/F1 for all **165 CV fits** and **27 validation
  trials**, checking row order, gold labels, class order, probabilities and hashes.
- Recomputed all **28 Bayesian contrasts** and **84 sensitivity rows** to numerical
  tolerance `1e-12` against the saved tables.
- Replayed **all 11 systems** from raw text on 32 fold-0 examples each, agreeing
  with saved probabilities within `1e-6`. Historical full reload checks remain saved.
- Synthetic checks matched fixed-training weights/probabilities against the earlier
  implementation, restored best validation weights, checked sparse boundaries,
  bundles/corruption detection, search resume, 22-fit tiny CV and actual timer expiry.
- Inspected figures. Steps 9/audit produced no test scores; the explicitly
  authorized frozen final fit/test and its additional checks are recorded below.

The audit found no numerical discrepancy at these tolerances. It does not replace
student review or additional generalization evidence. The Step 9 first-pass
linguistic review is recorded below.

## Step 9: held-out errors and frozen final recipe

Use split seed **0 only**: 60,005 unique source examples, each predicted out of
fold once by each of the 11 systems. The historical models use three fixed epochs.
These pooled descriptive scores differ slightly from the 15-fold means above;
there are no new independent repetitions or tuned CV scores here.

For the reference unigram MLP1, accuracy is **62.54%**, NLL **0.8461**, macro-F1
**0.6248**; there are **22,478** errors. Recall is 68.07% / 60.75% / 58.79% for
entailment / neutral / contradiction. The largest off-diagonal confusion is
**4,277 gold contradiction → predicted neutral** cases.

The regex-based text slices count distinct lowercase English word types:
hypothesis overlap is the fraction also present in the premise. Negation cues
are lexical markers in either sentence, not semantic scope analysis. Slice
memberships overlap and label distributions differ; these are descriptive rates.

| Slice | Rows | Unigram MLP1 accuracy | Bigram MLP1 accuracy | H-only logistic accuracy |
| --- | ---: | ---: | ---: | ---: |
| All rows | 60,005 | 62.54% | 55.47% | 61.08% |
| Negation cue | 1,552 | 68.36% | 60.12% | 69.39% |
| High overlap ≥.8 | 6,612 | 59.98% | 52.24% | 57.46% |
| High overlap, non-entailment | 2,094 | 45.27% | 39.49% | 45.51% |
| Low overlap ≤.2 | 10,182 | 64.28% | 56.46% | 63.61% |

The weak high-overlap/non-entailment score is consistent with difficulty
separating shared vocabulary from entailment, without establishing a causal
mechanism. Higher accuracy in the negation slice does not show reliable negation
reasoning: its label composition differs, and reviewed negation cases still fail.

Unigram MLP2 shares **20,281** anchor errors, corrects **2,197** and introduces
**2,055** where the anchor was correct. Pure-bigram MLP1 shares **15,996** errors,
corrects **6,482** and introduces **10,725**. H-only logistic corrects **3,671**
anchor errors and introduces **4,547**. Complementary errors do not by themselves
prove an ensemble would improve; none was fitted or scored.

### Fifty annotated cases

Deterministic, purposive sampling (seed 19): five per directed anchor confusion
(30), five confident shared-neural errors, five bigram-correct/anchor-wrong,
five H-only-correct/anchor-wrong and five anchor-correct/bigram-wrong. All 50
examples and exact premise groups are distinct. This gives diagnostic coverage,
not a random estimate of dataset errors or annotation noise.

AI-assisted notes cover 11 unsupported-detail cases, 11 attribute/action conflicts,
11 label/text uncertainties, seven paraphrase/abstraction cases, four commonsense
cases, three predicate/role cases and three contextual-negation cases. Categories
are descriptive reading judgements, not proven causes; students should review them.

- `train:324138`: background people support “not alone,” yet the anchor predicts
  contradiction with 97.7% confidence. `train:304491` similarly misses “not sleeping.”
- `train:340169`: bike versus horse riding changes a central object despite high
  overlap; the compared neural models wrongly predict entailment.
- `train:153352`: the passing car is an unsupported addition, which the anchor
  incorrectly treats as entailed.
- `train:99852` and `train:388831`: shaving/patient and donkey/pulling roles
  connect to predicate/argument structure. Unigram order loss is demonstrated
  separately in section 4; these examples do not isolate a prediction's cause.
- `train:520972`: “Nobody is standing” conflicts with a stated standing man,
  although the saved label is neutral. `train:210123` labels a marriage inference
  entailed from a restaurant conversation. These are flagged for review.

All **11 flagged cases retain their gold labels** and remain in metric denominators.
Confidence-targeted selection helps explain why questionable labels appear in
this review; the 11/50 fraction cannot estimate corpus-wide annotation noise.

Complete texts, predictions, selection reasons, flags and all 50 notes are in
[reviewed_cases.csv](results/project1/error_analysis/713ab1f5b3cb5801/split_seed_0/reviewed_cases.csv).
The same directory holds OOF summaries, class scores, confusions, text slices,
shared-error counts, category totals and a manifest with source/artifact hashes.
All implementation and annotation data appear literally in notebook section 9.

### Frozen final pipeline

[configs/snli_final.json](configs/snli_final.json) selects **5k case-preserving BPE,
unigram counts, separate sentence-role blocks, one 128-unit ReLU hidden layer**,
dropout .1, AdamW lr .001/weight decay .0001, batch 128, CPU/one thread and seed 0.

Choose this trade-off because unigram MLP1 has the lowest baseline mean NLL,
the extra-depth comparison is usually inside our primary ROPE, and the 8k
validation alternative adds only 0.09 pp with worse NLL and about 60% more
parameters (2.05M versus 1.28M). Indicator MLP2 retains the highest baseline mean
accuracy; this is a reasoned multi-criterion choice, not a posterior global ranking.

Freeze **five epochs**, the selected 5k candidate `29aa244710c5fabe`'s best
validation-accuracy checkpoint; no active ceiling flag. This respects accuracy
as the primary selection metric even though its minimum-validation-NLL epoch is
three. Its 67.74% validation score comes from the 100,001-row subset. Transferring
five epochs to the 559,164-row final pool is pragmatic, not confirmed optimal stopping;
the existing three-epoch CV results do not estimate this new final fit.

## Step 10: completed final fit and official test

Frozen recipe unchanged: 5k BPE, unigram counts, a 128-unit one-hidden-layer MLP,
seed 0 and five fixed epochs. Fresh preprocessing/weights use **559,164** original
train + validation rows after excluding test premises. This recovers 72 train
rows excluded by development's validation-overlap filter; reusing that CV pool
would have silently made the final training set too small.

Preparation, all five epochs and save/reload verification took **179.03
seconds (2.98 minutes)**, within 900 seconds. The fitted input dimension is
10,000 and the network has 1,280,515 parameters. Raw-text reload
agreement was 0 maximum absolute probability
difference on 32 training pairs before the first test inference.

| Final test metric | Value |
| --- | ---: |
| Rows | 9,824 |
| Accuracy | 74.58% |
| Macro-F1 | 0.7451 |
| NLL (natural log) | 0.6886 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| entailment | 0.7588 | 0.8026 | 0.7801 | 3,368 |
| neutral | 0.6960 | 0.6977 | 0.6969 | 3,219 |
| contradiction | 0.7835 | 0.7346 | 0.7583 | 3,237 |

Exactly one final-model probability pass was made on the cleaned official test.
Row IDs, gold labels, probability order, argmax and metric recomputation are
verified when loading. A process lock/journal protects the experiment; subsequent
Run All calls use saved predictions. The frozen JSON's `selection.test_scored=false`
is a historical statement about selection; the final manifest records completion.

### Graphs and interpretation

- [Learning curves](results/project1/final/snli_5k_unigram_mlp1/figures/learning_curves.png):
  online training NLL/accuracy over the five frozen epochs. Dropout and changing
  parameters make these different from fixed-model evaluation scores.
- [Test confusion matrices](results/project1/final/snli_5k_unigram_mlp1/figures/test_confusion.png):
  gold rows/predicted columns, raw counts and within-class rates.
- [Per-class metrics](results/project1/final/snli_5k_unigram_mlp1/figures/test_class_metrics.png):
  precision, recall and F1. Class support is listed above.
- [Confidence/reliability](results/project1/final/snli_5k_unigram_mlp1/figures/test_confidence.png):
  confidence distributions split by correctness and accuracy versus confidence in
  ten fixed equal-width bins. ECE is **0.0728**; mean confidence is **81.86%**.
  The curve lies below the diagonal in every nonempty bin: mean confidence
  81.86% exceeds observed accuracy 74.58%, indicating overconfidence. These
  describe saved predictions and are not a calibration fit.

The final larger-pool/five-epoch result cannot be read as an isolated tuning gain
against three-epoch subset CV or the 100k-subset validation search. It evaluates
the frozen selected pipeline once. No test-driven epoch, tokenizer, model,
threshold or calibration change was made. The error/representation/label limits
from Step 9 remain relevant; a strong argmax score alone does not establish full
semantic reasoning or calibrated uncertainty.

Artifacts: [final manifest](results/project1/final/snli_5k_unigram_mlp1/manifest.json),
exact training rows, full test predictions, learning history, complete prediction
bundle and the source notebook/package versions. Figures can be regenerated
without fitting or making further test predictions.

## Remaining work

Step 11: review all code and AI-assisted notes as a group, prepare the oral
explanations and up to six optional backup slides, and retain the repository URL,
sources and assistance disclosure for the later blog. See [the plan](PROJECT1_PLAN.md),
[course source map](COURSE_METHODS.md), [project instructions](project.ipynb)
and [AI disclosure](AI_ASSISTANCE.md).
