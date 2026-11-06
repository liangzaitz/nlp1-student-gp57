# Project 1 findings and decisions

## Current status

Steps 1–6 are complete and verified: workspace, dataset inspection, evaluation,
features, classifiers, and a measured SNLI pilot with all nine systems and two
baselines. The pilot selects 3 neural epochs, SAGA maximum 1,000 iterations,
and a 60,005-row complete-premise-group comparison pool. Its 15 fold pairs are
saved separately from the full-data reference. Reported CV comparisons and
final test scoring remain pending.

## Decisions established before experiments

| Decision | Motivation and source |
| --- | --- |
| SNLI, three labels, input is a premise/hypothesis pair | Group's selected task; metadata verified: 0 entailment, 1 neutral, 2 contradiction |
| Common BPE tokenizer | T1 segmentation method; fit on each training partition only |
| Indicators, unigram counts, unigram-plus-bigram counts | P1 representation coverage; CLASS PDF pages 20-24 and 49-50 |
| Log-linear, one-hidden-layer, two-hidden-layer classifiers | P1 architecture coverage; CLASS PDF pages 26-35 |
| Separate feature blocks for premise and hypothesis | Preserve which sentence supplied each feature |
| Accuracy as primary; NLL and macro-F1 as supporting metrics | CLASS PDF page 46; justify accuracy with observed class balance in Step 2 |
| Five premise-grouped folds, split seeds 0, 3, 5 | P1 CV requirement and our SNLI-specific adaptation |
| ROPE half-width 0.005 accuracy | Frozen before comparisons: gains within half a percentage point do not justify added complexity/cost on accuracy alone in this project |
| Pilot before the complete matrix | Measure computation cost before selecting final data size and training length |

The original `configs/snli_project1.json` retains the starting choices and unset
epoch count for reference/pilot reproduction. Step 6's separate
`configs/snli_project1_frozen.json` records the selected training configuration
and data budget for Step 7.

## Setup evidence

- Actual environment: Python 3.12.3, Linux x86_64, CPU PyTorch. The root course
  README specifies Python 3.14; this difference is recorded, not assumed away.
- The tracked `environment.txt` captures the initially verified package versions;
  each setup execution also writes its own local package snapshot and manifest.
- Dataset revision pinned in `configs/snli_project1.json`:
  `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`. Resolved with
  `HfApi.dataset_info("stanfordnlp/snli")` on 2026-11-05, without downloading rows.
- See `PROJECT1_PLAN.md` for completion checks, `COURSE_METHODS.md` for method
  sources/status, and the notebook for executed setup evidence.
- Verification: the CLI runs from the repository root, the notebook executes in
  a fresh kernel with working directory `part1`, and its bootstrap also finds
  the repository when launched from the root. CLI and notebook records agree on
  the interpreter, complete configuration, and configuration hash; both use
  `.venv`. Notebook validation and `pip check` pass.
- Generated data/results paths are Git-ignored. The setup manifest records the
  current dirty working tree rather than treating the initial commit as a
  complete record of these uncommitted changes.

## Findings to add as work progresses

- Evaluation and training-only preprocessing evidence: Steps 3-4 below.
- Model implementation/parameter evidence: Step 5 below; SNLI learning curves:
  Step 6.
- Paired CV metrics and posterior comparisons: Steps 7-8.
- Error categories and system-selection rationale: Step 9.
- Final test evaluation: Step 10.
- Oral notes and reproducibility summary: Step 11.

## Step 2: data inspection

Source: `stanfordnlp/snli` at revision
`cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`. These are measurements from the
executed data report, not expected counts copied from the dataset card.

### Cleaning and label balance

| Official split | Raw rows | Removed invalid labels | Removed unusable text after label filtering | Clean rows |
| --- | --- | --- | --- | --- |
| train | 550,152 | 785 | 0 | 549,367 |
| validation | 10,000 | 158 | 0 | 9,842 |
| test | 10,000 | 176 | 0 | 9,824 |

All 1,119 removed rows are unlabeled examples. No missing or whitespace-only
sentences were found. Original text, gold labels, ordering, and split membership
are retained. IDs such as `train:24750` refer to the original zero-based row,
including positions of removed rows; they are not reindexed after filtering.

| Training label | Count | Proportion |
| --- | --- | --- |
| entailment | 183,416 | 33.39% |
| neutral | 182,764 | 33.27% |
| contradiction | 183,187 | 33.35% |

This near balance supports accuracy as our primary decision metric. Macro-F1
and class reports will still reveal class-specific errors; NLL will assess the
probabilities rather than just the winning label. None of these model metrics
have been measured yet.

### Lengths and linguistic examples

Training premises have mean 12.85/median 12 whitespace-separated tokens;
hypotheses have mean 7.42/median 7. The respective 95th percentiles are 23 and
13, and maxima are 78 and 56. These counts use `str.split()` and are not BPE
lengths. They will differ from counts produced by other segmentation methods.

The notebook shows the first retained training row of each class. Rows
`train:2`, `train:0`, and `train:1` share a premise about a person jumping a
horse over a broken airplane. Their hypotheses respectively state being
outdoors/on a horse (entailment), training for a competition (neutral), and
being at a diner (contradiction). These are actual source examples; any pmf
used to explain argmax in the notebook is explicitly illustrative.

SNLI's English caption domain and elicited hypotheses limit how far later
results support claims about general language understanding. Hypothesis-only
shortcuts remain a later diagnostic, not a result established by this audit.

### Duplicates and shared premises

| Split | Unique premises | Repeated premise groups | Extra duplicate pair rows | Conflicting-label pair groups |
| --- | --- | --- | --- | --- |
| train | 150,736 | 150,736 | 626 | 64 |
| validation | 3,319 | 3,308 | 2 | 0 |
| test | 3,323 | 3,318 | 0 | 0 |

All 549,367 training rows belong to a repeated-premise group; the largest has
33 rows. The 626 extra duplicate training pair rows belong to 596 pair groups.
Sixty-four of these groups contain more than one gold label. We retain and
report them rather than silently selecting labels or deduplicating.

| Split pair | Shared unique premises | Rows affected in left split | Rows affected in right split | Shared exact pairs |
| --- | --- | --- | --- | --- |
| train / validation | 16 | 78 | 47 | 1 |
| train / test | 11 | 42 | 33 | 0 |
| validation / test | 1 | 3 | 3 | 0 |

Matching is by exact original strings, not normalized or semantic similarity.
This observed overlap means we must not assume the official splits are fully
premise-disjoint. Grouped CV controls sharing between CV folds, while the
official-holdout overlap requires an additional explicit policy in Step 3.
Keep the cleaned official splits as reference data and document any later
development-pool exclusions before fitting models. No test performance was
measured or used to make this observation.

### Reproduction and evidence

- Implementation: `snli_data.py`; run `python -m part1.snli_data` from the root.
- Executed notebook: section 2 of `practical_1.ipynb`.
- Prepared data: `data/snli/<revision>/cleaned-v1/`.
- Tables: `results/project1/metrics/data_audit/` (including all excluded IDs,
  duplicate-pair groups and IDs, exact overlap counts, and per-example lengths).
- Figures: `results/project1/figures/snli_class_balance.png` and
  `snli_training_lengths.png`.
- Data manifest: `results/project1/manifests/data_inspection.json`.
- Four offline tests cover filtering/row IDs/text preservation, exclusive
  removal reasons, label-schema mismatch, and distinct overlap/duplicate counts.
- The notebook executes successfully in a fresh kernel using cached source data.
- After reloading the saved DatasetDict, every retained premise, hypothesis,
  label, and original source-row identifier was checked against the pinned
  source. Label metadata, exclusion-count arithmetic, class proportions, and
  source fingerprints also agree.

The data and generated reports remain Git-ignored; the notebook's saved outputs
and this findings record preserve the main evidence in the repository.

## Step 3: evaluation protocol before training

### Pool policy and actual sizes

We preserve the cleaned official reference splits and derive selections using
exact original premise strings, giving the protected test priority. Holdout
labels/hypotheses are not read by the pool-selection helper. Premise hashes are
identifiers, not model features. This prevents exact-premise reuse; it does not
claim image-level or semantic independence.

| Selection | Rows | Unique premises | Policy |
| --- | --- | --- | --- |
| CV pool | 549,253 | 150,710 | Original train, excluding premises in either official holdout |
| Pilot validation | 9,839 | 3,318 | Original validation, excluding test premises |
| Final refitting | 559,164 | 154,028 | Original train + validation, excluding test premises |
| Final test | 9,824 | 3,323 | Entire cleaned official test, unchanged |

CV excludes 114 training rows: 72 whose premise occurs only in validation, 36
only in test, and 6 in both. These represent 26 unique training premises. The
pilot excludes 3 validation rows sharing a test premise. Final refitting excludes
42 original training rows and 3 validation rows; it may recover train/validation
shared rows because validation is no longer held out then. Every exclusion and
its pool-specific reason is saved with the original `example_id`.

### Saved folds and checks

Five shuffled `StratifiedGroupKFold` folds are saved for each seed `[0, 3, 5]`.
Every row receives one held-out assignment per repetition, with all rows of an
exact premise assigned together. Every training/held-out partition contains all
three classes, with zero shared premise groups. Across 15 held-out partitions,
sizes range from 109,848 to 109,857 rows. Their class proportions range as follows:

- Entailment: 33.3858%-33.3870%.
- Neutral: 33.2678%-33.2693%.
- Contradiction: 33.3443%-33.3455%.

Source: the generated `fold_summary.csv`; these are partition statistics, not
classifier scores. The default protocol ID is `0dc6821564b1d3f0`. Its directory
is `results/project1/folds/<protocol_id>/`, containing `protocol.json`,
`fold_assignments.parquet`, `pool_membership.parquet`, `excluded_rows.csv`,
`pool_summary.csv`, and `fold_summary.csv`. The latest pointer is
`results/project1/manifests/evaluation_design.json`.

All systems/baselines must use the saved `pool_index` order and fold columns.
Hashes detect changed artifacts, and reruns validate/reuse rather than replace
the folds. Training hyperparameter changes retain the assignments; a changed
pool, evaluation design, or splitter version receives a separate protocol ID.
Repeated folds share training data; neither fold scores nor repeated per-row
predictions should be treated as independent observations.

### Scores, practical equivalence, and selection policy

- Primary: accuracy on a 0-to-1 scale, motivated by observed near-balanced labels.
- Supporting: mean NLL in nats and macro-F1; also save per-class metrics/support
  and confusion matrices. NLL assesses probability assignments, while accuracy
  and F1 assess decisions. No model metrics have been measured yet.
- Probability columns and class reports always use `[0, 1, 2]`. Argmax ties go
  to the smallest label ID. Confusion matrices have gold rows/predicted columns.
  Undefined class metrics use zero. Probability-based NLL reporting uses float64
  machine-epsilon clipping, with original probabilities retained.
- ROPE: `[-0.005, +0.005]` accuracy. For this educational comparison, a change
  of at most half a percentage point is insufficient to favor a more complex
  or costly system on accuracy alone. This is our practical judgement, not an
  externally established application tolerance or an uncertainty interval.
- Tune a limited configuration set on eligible training groups/pilot validation;
  freeze epochs/settings before reported CV. Each outer training partition fits
  its own tokenizer/features/model. Outer held-out scores cannot choose that
  fold's checkpoint. CV may select the final design; the maximum selected CV
  score is not an unbiased final estimate.
- The pilot may require a fixed subset of complete groups. If so, create a
  separate documented protocol before comparisons; retain this full-pool record.
- Refit the selected frozen recipe using `final_refit`, then evaluate once on
  `final_test`; test performance cannot guide further tuning.

Implementation: `snli_evaluation.py`; notebook section 3 explains the policy,
displays computed exclusions/fold distributions, and reloads the saved folds.
Offline tests cover holdout priority without test targets, exact-string grouping,
row order/coverage, deliberate group leakage, seed reproducibility, immutable
reuse, changed-design IDs, and detection of modified artifacts.

## Step 4: training-only tokenization and sparse pair features

### Implementation and fit boundary

`snli_features.py` reuses `experiment.train_tokenizer`, with BPE vocabulary
target 5,000, minimum merge frequency 2, Metaspace `▁`/leading marker, and
`[UNK]`. No lowercasing, punctuation removal, stemming, or negation rewrite is
applied. Whitespace is handled by Metaspace. Unknown characters use `[UNK]`;
BPE units need not be words or morphemes. A vocabulary target need not equal
the actual size on small corpora; dimensions use the fitted size.

One shared tokenizer is fitted on premise then hypothesis of every outer
training row, retaining repeated text. Both roles contribute to the shared
within-sentence adjacent-token bigram inventory, with occurrence minimum 2,
cap 5,000, descending frequency, and ascending token-ID-pair ties. Held-out
transformation learns nothing new. No bigram crosses a role/row boundary.
Official validation/test sentences are not tokenized at this step.

For seed 0/fold 0, actual vocabulary and retained bigram sizes are both 5,000.
The extended coordinate layout is `[P unigrams; P bigrams; H unigrams; H bigrams]`.
Unigram spaces drop bigram blocks; indicators replace positive counts with 1.
There is no frequency normalization/TF-IDF weighting. All matrices use float32
CSR built directly from sparse coordinates, with no dense per-document vectors.

### Actual full-fold matrices

| Partition | Representation | Rows | Columns | Nonzero values | CSR bytes |
| --- | --- | --- | --- | --- | --- |
| Train | Indicators | 439,403 | 10,000 | 9,548,707 | 78,147,272 |
| Train | Unigram counts | 439,403 | 10,000 | 9,548,707 | 78,147,272 |
| Train | Unigram + bigram counts | 439,403 | 20,000 | 14,604,077 | 118,590,232 |
| Held-out | Indicators | 109,850 | 10,000 | 2,406,840 | 19,694,124 |
| Held-out | Unigram counts | 109,850 | 10,000 | 2,406,840 | 19,694,124 |
| Held-out | Unigram + bigram counts | 109,850 | 20,000 | 3,675,233 | 29,841,268 |

Source: executed feature command/notebook section 4. The extended training
matrix would require 35,152,240,000 bytes if allocated densely, versus
118,590,232 bytes for its CSR arrays. Dense size is calculated, not allocated;
CSR bytes do not include every preprocessing/runtime object or compressed files.
These are feature statistics, not classification results or pilot timings.

### Interpretation and reuse

Notebook section 4 segments actual training rows, checks repeated-token counts
versus presence, and examines constructed word-order examples using the actual
fitted tokenizer/inventory. It keeps sentence roles explicit, checks blank/
unknown inputs, verifies vocabulary immutability on transformation, and compares
features after encoder reload. Local bigrams add coordinates/capacity as well as
some order sensitivity; neither they nor extra MLP layers recover full syntax
or complete semantic roles from a lossy feature map. A retained-bigram cap can
also cause order-sensitive examples to remain indistinguishable.

With the actual fitted inventory, all three fixed notebook order examples
have identical indicator/unigram vectors and differ in two bigram coordinates.
For `the dog is chasing the man` versus `the man is chasing the dog`, the
retained premise features `('▁dog', '▁is')` and `('▁man', '▁is')` exchange counts.
These are constructed representation examples, not gold-labelled SNLI results.

Artifacts are under `results/project1/features/<protocol_id>/seed_0_fold_0/<feature_id>/`:
tokenizer/encoder JSON, original row IDs/labels/order, per-role ragged token
arrays, extended count matrices, and a manifest with configuration, versions,
source-code hashes, and artifact checksums. Unigram counts/indicators are derived
from the cached canonical counts; all classifier families can reuse them.
Feature settings/code/fold changes separate caches, while model-only settings
do not. Changed/missing files are rejected rather than silently overwritten.

Only seed 0/fold 0 has been materialized. The helper fits any other saved fold
on that fold's training rows when requested; the later CV runner will prepare
the remaining folds as needed. The pilot can still change the data budget or
feature hyperparameters before comparisons, using separately identified caches.
No classifier fitting or test scoring has occurred.

Eight additional offline tests cover independent hand-calculated feature
values, frequency/tie/cap rules, row/role boundaries, order collisions, role
swaps, unknown/blank input, batched/ragged encoding, actual vocabulary sizes,
fixed held-out transformation, encoder reloads, fold/config cache separation,
reuse, and deliberately modified artifacts.
All 18 project tests pass, and the complete notebook executes 20 code cells in
a fresh kernel, reusing the saved features with no error/widget outputs.

## Step 5: sparse classifiers, scoring, baselines, and prediction bundles

### Implemented models and capacity

`snli_models.py` adapts the supplied sparse SAGA logistic-regression approach to
precomputed CSR and reuses the tutorial `SparseMLP` first-layer forward pass
and initialization. The two-hidden-layer extension adds affine/ReLU/dropout
before the output. Both networks return log probabilities and train with
`NLLLoss`; no extra softmax is applied before that loss. Inputs remain sparse
through the first multiplication; hidden/output activations are dense.

Starting settings: SAGA with L2 regularization, `C=1`, tolerance `1e-4`, and
300 iterations; neural widths 128 / 128-64, dropout 0.1, AdamW learning rate
`1e-3`, weight decay `1e-4`, batch size 128. AdamW decay applies to all parameters
including biases. `num_epochs` remains unset until the SNLI pilot; a neural fit
rejects an unset count unless an explicit software-check override is supplied.
Initialization and batch-order seeds are recorded separately from CV seeds.

| Representation | Input dimension | Log-linear parameters | One-hidden-layer parameters | Two-hidden-layer parameters |
| --- | --- | --- | --- | --- |
| Indicators | 10,000 | 30,003 | 1,280,515 | 1,288,579 |
| Unigram counts | 10,000 | 30,003 | 1,280,515 | 1,288,579 |
| Unigram + bigram counts | 20,000 | 60,003 | 2,560,515 | 2,568,579 |

Source: section 4's actual dimensions and section 5's parameter formulas/actual
neural parameter counts. The first-layer sparse operation is also checked on
real feature rows using untrained networks; this does not fit SNLI weights.
These counts describe stored weights/biases, not minimal identifiable degrees
of freedom. Larger feature spaces increase capacity, while deeper models still
cannot distinguish identical feature vectors.

### Training and metric policy

Only training matrices/labels enter `fit_classifier`. Neural minibatches run
forward NLL, gradient clearing/backpropagation, and AdamW updates. The last
fixed epoch is retained; outer held-out labels cannot choose a checkpoint.
Histories contain online NLL/accuracy measured with changing weights and
training dropout. Ordinary prediction uses evaluation mode/no gradients.
Logistic fits retain iteration counts, convergence warnings, and a convergence
flag; hitting the iteration budget is not described as demonstrated convergence.

The scoring helper implements section 3's accuracy, NLL, macro-F1, fixed-order
class reports, and confusion matrices. Three columns always mean `[0,1,2]`,
ties select the first maximum, and probability-based NLL uses float64 epsilon
clipping. Tests compare controlled NLL values with PyTorch's log-softmax/NLLLoss.
No reported SNLI score has been generated at this step.

### Baselines and complete reloads

The majority baseline fits the training class-frequency distribution and uses
its argmax for decisions. Storing the distribution also allows NLL evaluation.
The configured hypothesis-only diagnostic is logistic/unigram counts on the
same folds, retaining only the hypothesis block. Shared training BPE/inventory
are held fixed to isolate removal of the premise coordinates. This is an input
ablation: preprocessing has seen the training premise corpus, while the
classifier receives no per-example premise features. Its prediction bundle
ignores premise content. Real diagnostic scores/conclusions remain pending.

Prediction bundles contain weights, tokenizer/bigram inventory, representation,
input mode/dimension, label names/order, settings/seeds, history/convergence,
versions/source hashes, context, and file checksums. Logistic uses joblib;
neural state dictionaries load onto an explicit device with weights-only
loading. Existing directories are not overwritten. Reloaded text-pair
probabilities match the originals, and changed preprocessing is rejected.

### Synthetic verification, distinct from the pilot

`python -m part1.snli_models` exercises all nine feature/model combinations,
class-prior and hypothesis-only baselines on 12 constructed red/green/blue
training pairs and three evaluation pairs. Targets are arbitrary software-check
IDs, not NLI annotations. Two neural epochs exercise optimizer/reload paths;
they do not select SNLI epochs. Only boolean checks, dimensions, parameters,
convergence, and reload differences are reported, without a performance ranking.
Runs are saved separately under `checkpoints/implementation_smoke/run_*/`.
Step 6 below records the actual measured SNLI pilot and frozen training budget.

Verification: all 29 project tests pass, including 11 new numerical/model/bundle
tests. The notebook completes all 24 code cells in a fresh kernel with no error/
widget outputs. The 11 synthetic bundles have normalized finite probabilities,
finite NLL, and zero maximum probability difference after reload. All four
synthetic SAGA fits report convergence. These observations verify software
behavior; they establish no SNLI classification performance or convergence.

## Step 6: measured SNLI pilot and frozen comparison budget

### Data, isolation, and saved evidence

Executed `python -m part1.snli_pilot` using `configs/snli_pilot.json`. Pilot ID
`c8206b35e66f6918` identifies the report in
`results/project1/pilot/c8206b35e66f6918/`; `manifests/pilot.json` points to it.
Sampling is label-independent, seed 0, and retains complete premise groups:

| Partition | Rows | Premise groups | Entailment | Neutral | Contradiction |
| --- | --- | --- | --- | --- | --- |
| Pilot training | 6,001 | 1,638 | 2,004 | 1,998 | 1,999 |
| Pilot validation | 1,202 | 406 | 421 | 392 | 389 |

Training uses the holdout-disjoint CV source pool; validation uses eligible
official validation rows. Original row IDs/order are saved. BPE and the capped
bigram inventory are fitted anew on these pilot training pairs only, yielding
5,000 unigram and 5,000 bigram types. The larger reference-fold tokenizer is
not reused for this pilot. Official test rows receive no predictions or scores.

All nine primary systems, the class-prior baseline, and the hypothesis-only
logistic/count diagnostic complete. Saved predictions contain gold/predicted
labels, original validation IDs and three fixed-order probabilities. Every
system has finite normalized probabilities, finite NLL, saved class/confusion
reports, and a complete text-pair prediction bundle. Maximum probability
difference after raw-text reload is exactly zero for all 11 models. Neural
probability NLL agrees with the log-softmax objective. These checks validate
the pipeline; pilot scores are tuning observations rather than CV estimates.

### Epochs, solver convergence, and interpretation

Six neural systems run through epoch 12, retaining candidates `[3,5,8,12]`.
The predeclared rule selects the earliest candidate whose mean pilot validation
accuracy across all six is within 0.005 of the best candidate mean:

| Epoch | Mean validation accuracy | Mean validation NLL |
| --- | --- | --- |
| **3, selected** | **0.568220** | **0.930163** |
| 5 | 0.562257 | 1.050958 |
| 8 | 0.564892 | 1.415466 |
| 12 | 0.558930 | 1.837565 |

The saved `learning_curves.png` shows decreasing training loss alongside rising
validation NLL at later epochs, with fluctuating validation accuracy. This is
evidence of overfitting on the small pilot, and demonstrates why accuracy alone
does not characterize probabilities. Three epochs are a common planning
choice under this rule, not proof of an optimum on a larger training pool.

All three primary SAGA fits hit the initial 300-iteration limit. Retrying with
ceiling 1,000 converges in 340 iterations (indicators), 365 (counts), and 385
(extended counts). The hypothesis-only fit converges in 257 under that ceiling.
The failed 300-iteration attempts are retained. Freeze `max_iter=1000`, `C=1`
and `tol=1e-4`; future fits must still record their own convergence status.

The class-prior baseline has pilot accuracy 0.350250 and NLL 1.098567; the
hypothesis-only diagnostic has accuracy 0.559068 and NLL 0.973893. Similar
hypothesis-only/full-input pilot accuracies motivate the planned paired CV
diagnostic; this small tuned pilot alone does not establish annotation-artifact
effects or a winning architecture/representation.

### Measured computation and selected comparison pool

The complete pilot takes 73.83 seconds including all 12-epoch neural searches,
solver retries, prediction/bundle checks, report generation and new fold checks.
Cold pilot preprocessing takes 0.983 seconds. Successful primary SAGA fits take
0.790–1.336 seconds; the selected first three neural training epochs take
1.312–2.537 seconds per system. The latter excludes validation callbacks and
checkpoint I/O. These are observations on this CPU under host load, not an
isolated benchmark or guaranteed CV runtime.

The largest sampled whole-process RSS is 1,158,266,880 bytes (about 1,105 MiB);
it includes imports/data and is not isolated model allocation. Extended training
CSR uses 1,647,968 bytes, versus 480,080,000 bytes for its dense float32 matrix.

Runtime planning scales selected nine-system training plus preprocessing
linearly with outer-training rows across 15 fold pairs, then applies a factor
of two. It excludes baselines, diagnostics and final refitting:

| Comparison target | Primary estimate | With factor-of-two allowance |
| --- | --- | --- |
| 15,000 | 7.19 minutes | 14.38 minutes |
| 30,000 | 14.38 minutes | 28.76 minutes |
| **60,000, selected** | **28.76 minutes** | **57.52 minutes** |
| Full eligible pool, 549,253 | 4.39 hours | 8.78 hours |

The largest proposed target within the two-hour planning budget is 60,000.
Whole-group selection actually yields **60,005 rows / 16,590 premise groups**,
with class counts 20,022 entailment / 19,969 neutral / 20,014 contradiction.
Sampling seed 0 is fixed; every system must use this same pool. The tiny
overshoot preserves the last complete group. Solver iteration counts, feature
frequencies, hardware load and I/O can make real CV costs differ from this
linear projection. Report the subset explicitly; it is not full-SNLI CV.

### Frozen recipe, new folds, and verification

`configs/snli_project1_frozen.json` records 3 neural epochs, one neural CPU
thread, SAGA ceiling 1,000, unchanged feature/architecture settings, and the
group-budget selection. The shared model helper restores the previous thread
count after each fit. Its optional epoch observer records independent pilot
validation curves without selecting outer-held-out checkpoints.

Protocol `3e3b8a39db826f1c` saves the 60,005-row pool and all 15 checked grouped
fold pairs. Held-out sizes are 11,998–12,010; every partition has all classes,
no train/held-out premise-group intersection, and each row is held out once
per repetition. All exclusions and original row mappings are saved. The full
reference protocol `0dc6821564b1d3f0` and its example feature cache remain
available. Step 7 must fit fresh preprocessing within each new training fold.
The pilot validation, final-refit, and test pool definitions remain as before.

All 35 project tests pass, including six new pilot/group-budget tests. All 29
notebook code cells execute in a fresh kernel with no error/widget outputs;
section 6 checks saved pilot artifacts and reloads the frozen folds without
retraining the pilot. No reported CV model comparisons have run; Step 7 is next.

## Sources

- `project.ipynb`: assignment requirements.
- `tutorial.ipynb` and supplied Python modules: implementation starting points.
- `COURSE_METHODS.md`: all four lecture decks with physical PDF page references.
- [SNLI dataset](https://huggingface.co/datasets/stanfordnlp/snli).
- [SNLI paper](https://aclanthology.org/D15-1075/).
- [scikit-learn StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- [scikit-learn metric definitions](https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss).
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/).
- [SciPy CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html).

Add sources when actually consulted and connect each reported finding to saved
outputs. Do not convert a planned hypothesis into an observed result.
