# Project 1 execution plan: SNLI

This is the working checklist for implementing Project 1 in small, reviewable
steps. The assignment requirements are in `project.ipynb`; explanations and
reusable examples are in `tutorial.ipynb` and the accompanying Python modules.
SNLI remains the dataset for Projects 2 and 3 and the eventual blog post.

The current course scope is the four lecture PDFs supplied by the group:
Introduction, How Language Works, Text Classification, and Text Generation,
together with tutorial 1. [COURSE_METHODS.md](COURSE_METHODS.md) records the
methods, exact source pages, their role in this project, and implementation
status. Lecture coverage does not automatically make a method a Project 1
requirement; the project notebook supplies the explicit assignment checklist.

## Current position

- [x] Read the project checklist, tutorial, and supporting modules.
- [x] Read all four supplied lecture decks and review their relevant diagrams;
  align this plan with the material covered so far.
- [x] Install the repository requirements into `.venv`, including CPU PyTorch.
- [x] Fix `TokenIndicator.__len__` to use `self.tokenizer` in the module and
  tutorial example; verify imports and encoding on a small synthetic example.
- [x] Complete Step 1: execute the setup command and initialized project notebook,
  record the environment/configuration, and establish documentation/output paths.
- [x] Complete Step 2: inspect the pinned dataset, save cleaned reference splits,
  figures and audits, and record observed duplicate/official-split overlap.
- [x] Implement Step 3: derive holdout-disjoint pools and save checked repeated
  grouped folds; freeze metrics, ROPE, and tuning/test policy before training.
- [x] Implement Step 4: fold-specific BPE and three sparse pair spaces; verify
  one complete outer fold and cache its preprocessing/counts for reuse.
- [x] Implement Step 5: sparse logistic/one- and two-hidden-layer MLPs, scoring,
  baselines, and full prediction bundles; verify with synthetic software checks.
- [x] Complete Step 6: run the measured SNLI pilot, freeze the training recipe
  and 60,005-row comparison pool, and verify its separate saved grouped folds.
- [ ] Implement and run the SNLI experiments below.

The existing environment uses Python 3.12.3; the course README specifies 3.14.
Record the actual environment and any departure from the course setup. The
earlier checks verified imports and selected helpers, not the full tutorial.
The Pylance typing diagnostics discussed in chat are separate cleanup work.

## Requirements and planned evidence

| Assignment requirement | Planned implementation | Evidence to save |
| --- | --- | --- |
| Understand the chosen task and data | SNLI loading, inspection, and statistics | Split/class counts, length plots, labelled examples |
| Explain text segmentation | Fold-specific BPE tokenizer | Configuration, tokenized examples, saved tokenizer |
| Indicators, bag-of-words, and an additional representation | Binary unigram indicators, raw unigram counts, unigram-plus-bigram counts | Feature definitions, order-sensitivity examples, and comparisons |
| Log-linear, single-hidden-layer MLP, and an additional architecture | Logistic regression, one-layer MLP, two-layer MLP | Architecture/configuration records and comparisons |
| Cross-validation and a relevant metric | Grouped five-fold CV; accuracy, with held-out NLL and macro-F1 as supporting metrics | Paired fold scores, probabilities, and held-out predictions |
| Bayesian estimation and a motivated ROPE | Bayesian correlated t comparison; initial ROPE +/-0.005 accuracy | Posterior plots and region probabilities |
| Prepare for oral questions | Explain all six project topics and tutorial concepts | Up to six optional backup slides and study notes |
| Prepare for the later blog | Reproducible code and documented sources/assistance | Run instructions, findings, citations, assistance log |

Running all nine representation/architecture combinations is our experimental
design, rather than an explicit requirement to run every possible combination.
Three repetitions of cross-validation, diagnostic baselines, error analysis,
and a final test evaluation strengthen the work beyond the minimum checklist.

## Scope decisions after reviewing the lectures

- Use unigram-plus-bigram counts as the third representation instead of TF-IDF.
  HC2a explicitly motivates multi-token features and n-grams (PDF pages 24 and
  49-50), and tutorial 1 discusses bag-of-bigrams. TF-IDF is a valid possible
  extension but is not explicitly developed in the supplied decks.
- Keep log-linear softmax regression, the tutorial's ReLU MLP, and a slightly
  deeper ReLU MLP. HC2a covers these building blocks and width/depth trade-offs
  (PDF pages 26-35); the extra architecture remains a small, explainable change.
- Describe every system as `sentence pair -> fixed features -> logits ->
  class probabilities -> argmax decision`. Evaluate the probabilities with
  held-out negative log-likelihood (NLL), and decisions with accuracy/F1,
  following HC2a PDF pages 8-9 and 46.
- Link error analysis to the distinction between lexical items, morphology,
  syntax, and meaning from How Language Works. BPE units are not guaranteed to
  be morphemes; bigrams preserve some local order but do not represent syntax.
- Retain cross-validation and Bayesian/ROPE analysis: these are explicit project
  requirements developed in tutorial 1, even though the lectures do not give the
  full Bayesian implementation.
- Treat grouped folds, sparse CSR storage, the full nine-system matrix, and the
  hypothesis-only diagnostic as our experimental/engineering choices, not extra
  methods claimed to be mandated by the slides.
- Keep embeddings, pooling, RNNs, autoregressive generation, and perplexity in
  the course-method tracker. They appear in HC2b, but SNLI label classification
  does not require a language model. GRU/LSTM details are explicitly deferred in
  HC2b to HC3b/tutorial 2; contextual pretrained encoders and LLM methods remain
  outside our core Project 1 implementation.
- Treat stemming, collocations, negation-scope features, uncertainty-aware
  decisions, and test-time dropout as discussed alternatives, not additional
  mandatory experiments. The required Bayesian comparison of CV scores is
  separate from Bayesian interpretations of test-time dropout.

## Intended organization

Create files as their phase needs them, rather than generating empty scaffolding.

```text
part1/
  PROJECT1_PLAN.md         # This checklist and decisions
  COURSE_METHODS.md        # Lecture/tutorial sources, method roles, and status
  practical_1.ipynb        # Task explanation, exploration, figures, interpretation
  project_setup.py        # Shared setup, seeds, and environment/run provenance
  README.md               # Setup and execution instructions
  environment.txt         # Initial verified package-version snapshot
  snli_data.py             # Load, clean, identify, and inspect examples
  snli_evaluation.py       # Derive development pools and save/reuse grouped folds
  snli_features.py         # Tokenization and sparse sentence-pair features
  snli_models.py           # Classifiers and matrix-based training helpers
  snli_pilot.py            # Measured pilot, epoch/budget selection, saved report
  snli_experiments.py      # Planned controlled CV/final experiment runner
  configs/                # JSON configuration files
  results/project1/       # Metrics, predictions, manifests, checkpoints, figures
  PROJECT1_FINDINGS.md     # Motivations, observations, limitations, oral notes
  AI_ASSISTANCE.md         # Assistant usage and how its output was checked
```

Keep explanations in the notebook and reusable logic in Python modules. Reuse
the supplied code where appropriate. Restrict changes to tutorial modules to
necessary fixes or clearly justified adaptations. Keep dataset caches and large
generated artifacts out of Git, while preserving instructions to recreate them.

## Step 1: Establish the project workspace and reproducibility records

- [x] Initialize `practical_1.ipynb` as a valid notebook with sections matching
  the steps below and executed setup cells.
- [x] Confirm the notebook and command-line setup use the same `.venv` Python.
- [x] Record the Python/package versions, repository commit, hardware/device,
  dataset revision, and all random seeds in an experiment manifest.
- [x] Establish output paths and configuration names.
- [x] Start the findings and AI-assistance records. Include assistance with the
  plan, dependency setup, tokenizer correction, and course-material review
  already performed.
- [x] Add source references in notebook explanations and update the method
  tracker. Continue updating these as later concepts are implemented/verified.

Done when: the notebook opens and imports the project, the environment can be
recreated, and experiment outputs have a documented home.

Completed evidence: `project_setup.py`, `configs/snli_project1.json`, the executed
setup cells in `practical_1.ipynb`, `environment.txt`, `README.md`,
`PROJECT1_FINDINGS.md`, and `AI_ASSISTANCE.md`. The latest local manifest is
`results/project1/manifests/setup.json`. Checked CLI/notebook interpreter and
configuration agreement, root/part1 path handling, notebook validity, and
dependency consistency. Configuration remains provisional until the pilot.

## Step 2: Load, clean, and understand SNLI

- [x] Load `stanfordnlp/snli` using Hugging Face Datasets and record its revision.
- [x] Preserve the official training, validation, and test splits.
- [x] Verify the label mapping from dataset metadata: 0 entailment, 1 neutral,
  2 contradiction. Entailment means the premise supports the hypothesis.
- [x] Filter labels outside `{0, 1, 2}`, including `-1`, and record removals.
- [x] Keep stable identifiers using the original split and row index.
- [x] Summarize split sizes, label counts/proportions, text lengths, missing or
  empty text, duplicate pairs, and repeated premises.
- [x] Audit premise/pair overlap between splits without using test performance
  to guide development. Document any cleaning policy before comparisons.
- [x] Inspect illustrative training examples from all three classes; selection
  is the first retained row of each class, rather than a representative sample.
- [x] Introduce the course notation `x = (premise, hypothesis)`, `K = 3`,
  `p_theta(y | x)`, and `y_hat = argmax_k p_theta(k | x)`; distinguish the
  probabilistic model from its decision rule (HC2a PDF pages 8-9).
- [x] Save class/length plots and a short explanation of neutral versus
  contradiction, caption-domain limitations, and possible annotation artifacts.

Done when: the notebook explains the task, shows actual cleaned counts, and
preserves traceability from every example back to its source row.

Completed evidence: `snli_data.py`, executed notebook section 2, four offline
data-audit tests, the findings record, and local prepared data/CSV/PNG/manifest
artifacts. Clean sizes: 549,367 train; 9,842 validation; 9,824 test. The audit
found 16 train/validation shared premises, 11 train/test, and 1 validation/test;
Step 3 must address this instead of assuming official premise disjointness.

## Step 3: Freeze the evaluation design

- [x] Set the official training split as the source of reported CV folds.
- [x] Reserve official validation for development checks or limited pilot tuning;
  keep the official test set untouched by model selection and error-driven tuning.
- [x] Define a separate leakage-aware development pool using the Step 2 overlap
  audit. Decide/document exclusions for shared official-holdout premises before
  fitting any model; preserve the cleaned official reference splits and test
  examples. Record affected source IDs rather than silently editing the dataset.
- [x] Group by premise text so examples sharing a premise stay in one fold.
- [x] Use `StratifiedGroupKFold` with approximately balanced fold sizes/classes.
- [x] Plan five folds and split seeds `[0, 3, 5]`, saving all assignments once.
- [x] Check that premise groups are disjoint across training/evaluation partitions
  and that each row receives one held-out prediction per repetition.
- [x] Establish checked reuse of exact assignments and row order for every system,
  including baselines; model execution follows in Step 7.
- [x] Choose accuracy as the primary score, justified using observed class balance.
  Also record macro-F1, class metrics, and confusion matrices for decisions, and
  mean held-out NLL for predicted probabilities (HC2a PDF page 46).
- [x] Define the shared argmax/tie-breaking convention for all classifiers.
  Keep uncertainty-aware decision rules as optional discussion (HC2a PDF pages
  51-54), rather than adding a selective-prediction or AUROC experiment.
- [x] Define the initial ROPE as +/-0.005 on the 0-to-1 accuracy scale: half a
  percentage point in either direction. Write down why that difference is small
  enough to count as practically equivalent for this project.

Done when: saved folds, primary metric, ROPE, and tuning policy are documented
before examining the reported comparisons.

Implementation: `snli_evaluation.py`, notebook section 3, and offline leakage/
coverage/reuse tests. CV excludes original training rows whose exact premise
appears in either official holdout. Pilot validation excludes test premises.
Final refitting may recover train/validation overlaps by using original cleaned
train + validation, still excluding all test premises. Reference splits/test
rows are unchanged; every exclusion is recorded by original ID. Saved folds
have checksums and are reused for the same data/design. All metrics are still
unmeasured. The Step 6 pilot may require a separately identified group subset;
it must not silently replace these full-pool assignments.

## Step 4: Implement tokenization and sentence-pair features

- [x] Reuse the Hugging Face BPE training helper, initially targeting vocabulary
  size 5,000, minimum frequency 2, and Metaspace pretokenization.
- [x] Implement fitting/saving a shared tokenizer on each requested fold's
  training premises/hypotheses only; verify on seed 0/fold 0. Other folds fit
  their own preprocessing on demand during the later CV runs.
- [x] Keep casing/punctuation policy explicit and consistent. Preserve negation.
  Explain potential information loss from normalization (HC2a PDF page 24).
  Start without stemming or special negation-scope rules so changes remain
  attributable to the planned feature comparison.
- [x] Encode the two sentence roles separately and concatenate their vectors:
  `h(premise, hypothesis) = [v(premise); v(hypothesis)]`.
- [x] Implement binary token indicators and raw token counts using the same
  vocabulary. Input dimensionality is twice the actual vocabulary size.
- [x] Implement unigram-plus-bigram counts as the third representation. A bigram
  is an ordered pair of adjacent BPE tokens, counted within each sentence only;
  do not create features across the premise/hypothesis boundary. This extends
  the lecture's multi-token features using the tutorial's BPE segmentation.
- [x] Learn the bigram feature inventory on the fold's training rows only. Start
  with minimum frequency 2 and a maximum of 5,000 bigram features, then fix the
  cap during the pilot. These numbers are our hyperparameters, not lecture rules.
  With `V` unigram and `B` bigram features, pair dimensionality is `2 * (V + B)`.
- [x] Demonstrate a pair of sentences with equal unigram counts but different
  meanings/order, and inspect how bigram features change. Explain remaining
  long-range/syntactic limitations and the extra dimensionality/capacity.
- [x] Build CSR matrices directly, using float32 where appropriate. Avoid the
  supplied trainers' full lists of dense vectors before sparse conversion.
- [x] Cache fold-specific tokenization/counts for reuse across representations
  and classifiers without mixing preprocessing between folds.
- [x] Check dimensions, binary/count behavior, unknown/empty inputs, role order,
  and consistent transformation of held-out data.

Done when: all three representations work on sentence pairs, use sparse storage,
and fit no preprocessing on the outer evaluation fold or test set. The notebook
explains what each feature space preserves and discards. Bigram classification
features are not the same model as a bigram language model from HC2b.

Implementation: `snli_features.py`, notebook section 4, and offline feature/
fit-boundary/cache tests. Seed 0/fold 0 is materialized in full: 439,403 training
and 109,850 held-out rows; actual `V=5,000`, `B=5,000`, so pair dimensions are
10,000 / 10,000 / 20,000. Only that fold has been fitted at this stage; the
helper prepares other folds on demand without sharing learned preprocessing.
Canonical extended counts and ragged token arrays are cached; unigram counts
and indicators derive from the same counts. No official validation/test text
is tokenized and no classifier is trained. Settings remain provisional pending
the Step 6 pilot; a smaller data pool requires its own protocol/cache.

## Step 5: Implement models and baselines

- [x] Implement multinomial logistic regression using the supplied approach,
  accepting sparse feature matrices and recording convergence status.
- [x] Reuse/adapt the supplied sparse-input one-hidden-layer MLP: initially
  128 hidden units, ReLU, and dropout 0.1.
- [x] Add a two-hidden-layer MLP: initially 128 then 64 units, ReLU and dropout.
- [x] Initially use AdamW, learning rate 1e-3, weight decay 1e-4, and batch size
  128 for neural models; treat these as starting configurations, not findings.
- [x] Explain logits/softmax and conditional maximum-likelihood estimation,
  using average loss `-mean(log p_theta(y | x))` (HC2a PDF pages 26-44).
  Reuse `LogSoftmax` with `NLLLoss` from the tutorial, or raw logits with
  `CrossEntropyLoss`; do not apply softmax twice.
- [x] Explain mini-batch gradients, automatic differentiation, parameter updates,
  dropout, and weight decay. Disable dropout during ordinary evaluation. AdamW
  is the tutorial's optimizer choice; the lectures discuss optimizers generally,
  without prescribing our learning rate or a specific AdamW configuration.
- [x] Record input dimensionality and model parameter counts. Show why changing
  feature inventory, hidden width, or depth changes capacity, and why an MLP
  cannot distinguish inputs that the feature function maps to the same vector.
- [x] Seed Python, NumPy, classifier randomness, and PyTorch; record split seeds
  separately from model initialization seeds.
- [x] Support saving/reloading the model together with the correct tokenizer,
  feature transformation, label mapping, and configuration.
- [x] Implement a majority-class baseline to fit each training partition.
  If evaluating its NLL too, use the training class-frequency distribution for
  probabilities and its argmax as the majority-class decision.
- [x] Plan/implement a hypothesis-only diagnostic with one fixed configuration on
  the same folds, to investigate how much evidence the premise contributes.

Checkpoint selection policy: use a fixed epoch count set before the reported
CV runs, or select epochs using an inner grouped training split. The outer held-out
fold must not choose the checkpoint whose score is then reported. If tuning
within a fold, fit learned preprocessing on the inner training portion too;
after selection, refit on the outer training partition as appropriate.

Done when: all three model families accept all three representations, have
documented training settings, and reproduce predictions after reloading.

Implementation: `snli_models.py`, notebook section 5, and numerical/bundle tests.
The single-hidden-layer model inherits the tutorial SparseMLP forward pass;
the second architecture adds another hidden affine/ReLU/dropout stage. Training
uses CSR batches, LogSoftmax/NLLLoss, and a fixed final epoch, with no outer
held-out checkpoint selection. SAGA records convergence/iterations; starting
settings are `C=1`, `max_iter=300`, `tol=1e-4`. The reference configuration keeps
neural epochs unset; Step 6 saves a separate frozen recipe. Step 5's synthetic
checks fit all nine systems, class-prior and hypothesis-only baselines, and
reproduce saved text-pair probabilities. The hypothesis-only diagnostic uses
logistic/unigram counts with the same training tokenizer/inventory, dropping
premise coordinates; it is a
classifier-input ablation rather than premise-free preprocessing.

## Step 6: Run a pilot and establish the computation budget

- [x] Run an end-to-end pilot on a small fixed training sample using complete
  premise groups. Keep pilot results distinguishable from reported CV results.
- [x] Check loss curves, convergence, prediction labels, sparse memory use,
  checkpoint reloads, and the full metrics/predictions output format. Verify
  that probabilities are finite, have three entries, and sum to one, and that
  held-out NLL agrees with the same cross-entropy objective used in training.
- [x] Time preprocessing and training for each model family before scaling up.
- [x] Choose the fixed training length and finalize the small configuration set.
- [x] Decide whether the complete matrix is feasible on full SNLI. If necessary,
  select one reproducible subset by complete premise groups, approximately
  30,000-60,000 examples as an initial budget target. Record the exact size and
  class distribution, and use that same subset for every comparison.
- [x] Recreate and save final folds if the chosen data pool changes after the pilot.
- [x] Freeze configurations and data selection before the reported runs.

Done when: a complete small run works and measured runtime/memory support the
chosen dataset size. Subsampling is a documented budget choice, not a result to
hide; do not extrapolate its scores to training on the full dataset.

Completed evidence: `snli_pilot.py`, `configs/snli_pilot.json`, notebook section 6,
and report `results/project1/pilot/c8206b35e66f6918/`. All nine systems and two
baselines complete on 6,001 training / 1,202 independent validation pairs,
with fresh training-only preprocessing, finite normalized probabilities,
matching NLL objectives, and exact saved-bundle prediction reloads.
The predeclared epoch rule selects 3 epochs; the three primary SAGA fits require
a 1,000-iteration ceiling after hitting 300. The frozen recipe is
`configs/snli_project1_frozen.json`, with one neural CPU thread and unchanged
feature/architecture settings. The reference/pilot configuration is retained.

Linear runtime planning with a factor-of-two allowance estimates about 58
minutes for 135 primary fits and fold preprocessing on a 60,000-row target,
versus about 8.8 hours for the eligible full pool. These are estimates; baselines,
diagnostics and final refitting are additional. Selecting complete groups gives
60,005 comparison rows / 16,590 premise groups, class counts 20,022 / 19,969 /
20,014. Protocol `3e3b8a39db826f1c` contains all 15 checked fold pairs, held-out
sizes 11,998–12,010. The full reference protocol remains available separately.
All 35 project tests pass; all 29 notebook code cells execute in a fresh kernel.
No reported CV model comparisons or test scores have been produced.

## Step 7: Run the controlled comparison

| Representation | Logistic regression | One-hidden-layer MLP | Two-hidden-layer MLP |
| --- | --- | --- | --- |
| Indicators | Run | Run | Run |
| Raw counts | Run | Run | Run |
| Unigram + bigram counts | Run | Run | Run |

- [ ] Run all nine designs on the saved folds with their frozen configurations.
  Five folds times three repetitions gives 135 primary fits; pilots, tuning,
  baselines, diagnostics, and final refitting are additional work.
- [ ] Support resuming completed work and distinguish failed/incomplete runs.
- [ ] Save one metrics row per system/repetition/fold, with accuracy, held-out
  NLL, macro-F1, runtime, feature/model size, data/configuration identifiers,
  seed, and checkpoint information.
- [ ] Save held-out predictions keyed by original example ID, with gold/predicted
  labels and class probabilities for all primary systems.
- [ ] Run the majority-class baseline and hypothesis-only diagnostic using the
  same folds and paired scoring conventions.
- [ ] Produce summary tables, fold-score plots, and learning curves.
- [ ] Interpret accuracy and NLL separately: assigning more probability to the
  correct label need not change the argmax decision, and a high accuracy alone
  does not characterize the entire predicted pmf (HC2a PDF pages 46 and 51-55).

Done when: every planned system has matching completed folds and a traceable
score/prediction record. Report variability across folds as descriptive evidence;
do not treat all folds or repeated predictions as independent measurements.

## Step 8: Compare systems with Bayesian estimation

- [ ] Assemble paired score arrays in identical `(repetition, fold)` order.
- [ ] Use the tutorial's Bayesian correlated t model with `runs=3` for three
  repetitions of five-fold CV. Reinitializing on unchanged folds is not a new
  cross-validation partition.
- [ ] Compare representations at a fixed architecture and architectures at a
  fixed representation; prioritize these planned contrasts in the write-up.
- [ ] Use the preselected ROPE; summarize the posterior mean difference and
  uncertainty, plus `P(A better)`, `P(practically equivalent)`, `P(B better)`.
- [ ] Save posterior plots with the difference direction clearly labelled.
- [ ] Explain overlap between training folds, the approximate correlation
  correction, and the assumptions/limits of the analysis.
- [ ] Connect repeated experiments to HC2a PDF page 47: fixing a seed documents
  a run but does not remove experimental variation or demonstrate replication.
- [ ] Optional: inspect sensitivity to alternative reasonable ROPE sizes,
  keeping the original threshold identifiable rather than choosing a favorable one.

Done when: conclusions quantify practical differences and uncertainty instead of
declaring a winner solely from the largest average accuracy.

## Step 9: Analyze held-out errors and select a design

- [ ] Generate confusion matrices and class metrics for representative systems.
- [ ] Inspect approximately 30-50 errors from saved CV/validation predictions,
  keeping example identifiers and the selection procedure.
- [ ] Categorize patterns such as negation, lexical overlap, sentence-role/word-order
  effects, neutral/contradiction confusion, world knowledge, and ambiguous labels.
- [ ] Compare shared failures and examples where two systems disagree.
- [ ] Interpret the hypothesis-only diagnostic in relation to annotation artifacts.
- [ ] Explain the limitations of bag-based encodings and what extra MLP layers
  can/cannot recover from information already discarded by the features.
- [ ] Check whether bigrams help the local-order/negation examples that motivated
  them, without claiming they solve hierarchy, semantic roles, or long-range
  dependencies (How Language Works PDF pages 20-35; HC2a PDF pages 49-50).
- [ ] Discuss ambiguous gold labels and distinguish maximum predicted probability
  from evidence of human agreement or truth (HC2a PDF pages 51-54).
- [ ] Select a design using CV performance, Bayesian evidence, runtime, memory,
  and simplicity. Document the reasons and any remaining uncertainty.

Done when: the selected design has an evidence-backed rationale and limitations,
with no test-driven revisions.

## Step 10: Refit and evaluate once on the official test set

- [ ] Freeze the selected configuration, preprocessing policy, and training length.
- [ ] Refit on the available non-test development data. If combining official
  train and validation, document that choice and refit the tokenizer and the
  unigram/bigram feature inventory there. Use Step 3's `final_refit` selection:
  original cleaned train + validation excluding every exact test premise.
- [ ] Evaluate the selected pipeline once on the cleaned official test split.
- [ ] Save final probabilities/predictions, accuracy, NLL, macro-F1, class report,
  and confusion matrix. Language-model perplexity is not our SNLI metric.
- [ ] Keep final test results distinct from CV results. Do not retune based on
  the test outcome or claim that a single test score reproduces the CV posterior.

Done when: the final artifact can be loaded to predict a sentence pair and its
held-out test performance is recorded with the exact training recipe.

## Step 11: Package the work and prepare for the oral assessment

- [ ] Check the notebook runs in order from a fresh kernel after documented setup.
  Long training cells may load documented saved artifacts for inspection, with
  separate instructions for reproducing the training itself.
- [ ] Document commands/configurations to recreate cleaning, folds, training,
  analysis, figures, and final evaluation.
- [ ] Finish the findings, references, environment snapshot, and AI-assistance log.
- [ ] Prepare up to six backup slides: data/task; tokenization; representations;
  architectures; evaluation; findings/selection/limitations. Include the project
  repository URL. The project notebook allows these as optional aids; the
  Introduction overview describes a handful of slides plus the repository URL.
  Preparing the compact deck and URL covers both descriptions.
- [ ] Review tutorial questions: Unicode and segmentation, BPE merges/unknowns,
  Heaps' law, sparse features, softmax, probability versus decision rules,
  overfitting/dropout, cross-validation, and Bayesian/ROPE interpretation.
- [ ] Use `COURSE_METHODS.md` to connect these explanations to the four lectures:
  morphology versus BPE, order versus counts, fixed versus learned features,
  log-linear versus neural logits, MLE/NLL versus accuracy, and the limits of
  argmax. Review HC2b concepts without adding a generation implementation.
- [ ] Make sure every group member can explain the design choices, calculations,
  alternatives, failures, and assistance used. Rehearse concise answers for the
  15-minute oral assessment described in the Introduction, PDF page 8.
- [ ] Check the course's official deadline/oral schedule separately; dates are
  not established by this repository.

Done when: every assignment checklist item has saved evidence and the group can
explain it. A full blog post is a later course deliverable; preserve its materials.

## How we will work through the plan

Implement one numbered step at a time when requested. For each step, describe the
change, perform checks appropriate to it, update this checklist and findings,
update the method tracker with implemented/verified evidence, and explain the
concept needed for the oral assessment. Keep the four supplied decks and tutorial
1 as the current scope unless the group provides later material. Do not
automatically start the complete training matrix while still establishing the
pilot budget.

Steps 1–6 are complete and verified. Next is Step 7: a resumable paired CV
runner using `configs/snli_project1_frozen.json` and the 60,005-row protocol.
The measured pilot is tuning evidence; reported CV comparisons remain pending.

## Starting references

- Local assignment: `project.ipynb`.
- Lecture/tutorial method map and source-page index: `COURSE_METHODS.md`.
- Local explanations: `tutorial.ipynb`, `bpe.py`, `clslib.py`, `experiment.py`,
  `loglin.py`, `smlp.py`, and `analysis.py`.
- SNLI dataset: https://huggingface.co/datasets/stanfordnlp/snli
- SNLI paper: https://aclanthology.org/D15-1075/
- Annotation artifacts: https://aclanthology.org/N18-2017/
- Grouped CV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
- Bayesian comparison: https://baycomp.readthedocs.io/en/latest/functions.html

These references identify starting points; record additional sources actually
consulted during implementation and interpretation.
