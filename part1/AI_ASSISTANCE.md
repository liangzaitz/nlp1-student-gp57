# AI-assistance record

This records assistance supplied by Codex during this project. Group members
should add their own review/learning notes and later assistance, and summarize
the relevant use in the eventual blog. Assistant verification does not establish
that every student has reviewed or can explain the implementation.

## 2026-11-05: Repository review and initial planning

- Assistance: read the repository, Project 1 checklist, tutorial, and SNLI
  documentation; propose a phased experiment plan.
- Outcome: `PROJECT1_PLAN.md` with required comparisons, CV, Bayesian/ROPE
  analysis, reproducibility, and oral preparation.
- Checks: compared the plan with the actual assignment checklist; distinguished
  required coverage from our proposed full nine-system comparison.

## 2026-11-05: Environment diagnosis and tokenizer repair

- Assistance: diagnose reported Pylance messages and distinguish static typing
  issues from runtime failures; install missing repository dependencies.
- Code changed: `TokenIndicator.__len__` now references `self.tokenizer` in
  `clslib.py` and the matching tutorial example.
- Checks: import PyTorch and `part1`, verify token-indicator length/encoding,
  and run `pip check`. Exercise the existing Bayesian helper on synthetic scores.
- Scope: the Pylance-only changes discussed in chat were not applied to
  `analysis.py` or `experiment.py`.

## 2026-11-05: Review of current course material

- Assistance: read all four supplied lecture PDFs and inspect relevant diagrams;
  map methods and page references in `COURSE_METHODS.md`.
- Outcome: change the proposed third representation from TF-IDF to
  unigram-plus-bigram counts, add held-out NLL, and connect explanations to the
  lecture's probability/decision distinction and linguistic limitations.
- Checks: preserve the P1 requirements and distinguish lecture coverage,
  optional extensions, and our task-specific engineering choices.

## 2026-11-05: Step 1 implementation

- Assistance: initialize `practical_1.ipynb`, implement `project_setup.py`, add
  the shared JSON configuration, output conventions, setup instructions,
  environment snapshot, and initial findings/disclosure records.
- Data access: resolve the public SNLI repository's immutable revision through
  Hugging Face's metadata API. No SNLI rows or labels are inspected at this step.
- Checks: validate the notebook and configuration, run the setup command and
  notebook in the project virtual environment, compare their interpreter and
  configuration records, and check dependency consistency.
- Student review to add: explain the configuration, the purpose of recording
  versions/seeds/Git state, and the difference between reproducibility and
  experimental variation. Record any changes made after reviewing the output.

## 2026-11-05: Step 2 implementation and data audit

- Assistance: implement `snli_data.py`, fill notebook section 2, generate figures
  and CSV reports, and record actual dataset counts and observations.
- Data handling: download the pinned public SNLI revision, verify its label
  mapping, exclude invalid labels/unusable text, preserve source row identifiers
  and official splits, and save the cleaned DatasetDict locally.
- Audit: measure class/length distributions, repeated premises, duplicate pairs,
  conflicting-label pairs, and exact cross-split overlap. Held-out splits are
  inspected structurally; no test predictions or model comparisons are made.
- Checks: four synthetic offline tests exercise provenance and audit edge cases;
  run the real-data command and execute the notebook in a fresh kernel using
  cached data. Inspect the generated figures and prepared-data round trip.
- Important finding to review: official splits contain some exact shared
  premises. Document a separate leakage-aware evaluation policy in Step 3;
  the Step 2 cleaned reference splits retain this overlap and duplicate pairs.
- Student review to add: explain neutral versus contradiction, verify the
  displayed training examples and source IDs, justify the primary metric using
  class balance, and distinguish repeated premises from repeated full pairs.

## 2026-11-05: Clarify cached loading and notebook output

- Assistance: distinguish successful cached loading from a download failure;
  confirm the saved DatasetDict contains sentence pairs and the expected counts.
- Presentation changed: disable Hugging Face progress bars in the notebook,
  remove saved widget state, and display an actual loaded training row in 2.2.
  This leaves the source revision and cleaning logic unchanged.
- Checks: execute all 11 code cells in a fresh kernel with offline mode unset;
  confirm there are no error outputs, saved widget views, or offline warnings.

## 2026-11-05: Step 3 evaluation design and saved grouped folds

- Assistance: implement `snli_evaluation.py`, update configuration and notebook
  section 3, and document the overlap-exclusion, tuning, scoring, and ROPE policies.
- Data handling: preserve cleaned reference splits; exclude training premises
  shared with either official holdout from CV, and exclude test premises from
  pilot validation/final refitting. Only test premises/source IDs enter selection.
  Save every exclusion, selection membership, and repeated five-fold assignment.
- Reproducibility: identify protocols by source fingerprints/design/version;
  validate and reuse saved folds with checksums. Training hyperparameters do not
  change assignments. A changed design gets a separate protocol directory.
- Checks: synthetic tests deliberately introduce leakage, wrong/missing/reordered
  rows, unassigned folds, and modified files; verify deterministic seeds and
  immutable reuse. Generate all 15 real-data partitions and check class presence,
  complete coverage, original row alignment, and exact-premise separation.
- Notebook validation: all 15 code cells complete in a fresh kernel; section 3
  reuses the same saved full-data protocol, with no error/widget outputs.
- Student review to add: explain why grouping differs from ordinary random
  folds, why the 114 training exclusions differ from the 1,119 invalid-label
  removals, and why a ROPE is a practical judgement rather than an error bar.
  Explain the final-refit pool, the pilot's role, and why no test-driven tuning
  is permitted. No tokenizer/classifier has been trained at this step.

## 2026-11-05: Step 4 BPE and sparse sentence-pair features

- Assistance: implement `snli_features.py`, reuse the tutorial BPE helper, add
  the feature artifact directory/configuration policy, and complete notebook
  section 4 with actual tokenization, sparse-size, count, and order examples.
- Fit boundary: fit shared BPE and capped within-sentence BPE bigram inventory
  only on an outer fold's training premises/hypotheses. Transform held-out rows
  using fixed coordinates. Official validation/test text is not tokenized.
- Representation: separate sentence-role blocks for binary indicators, raw
  unigram counts, and raw unigram-plus-bigram counts. Construct float32 CSR
  directly; cache ragged tokenization/canonical counts once for model reuse.
- Materialization: prepare seed 0/fold 0 in full (439,403 train / 109,850 held-out
  pairs). Other folds fit separately on demand; no classifier/pilot matrix runs.
- Checks: eight new offline tests check hand-calculated values, sentence/role
  boundaries, frequency/cap/tie rules, order/role changes, blank/unknown input,
  actual versus target vocabulary sizes, held-out vocabulary immutability,
  serialization, cache reuse/separation, and detection of modified artifacts.
- Verification: all 18 project tests pass. All 20 notebook code cells execute
  in a fresh kernel with no error/widget outputs, reusing the saved full-fold
  features. Each of the three constructed order pairs collides in unigram/
  indicator space and changes two retained bigram coordinates.
- Student review to add: explain BPE versus linguistic morphology, raw counts
  versus presence, the two sentence-role blocks, why a bigram inventory must be
  training-only, how the cap affects order sensitivity/capacity, and why sparse
  storage is needed. Distinguish BPE merge pairs, classifier bigram features,
  and a bigram language model. Classifier comparisons remain pending.

## 2026-11-05: Step 5 classifiers, baselines, and complete model reloads

- Assistance: implement `snli_models.py`, add explicit logistic/diagnostic
  configuration, and complete notebook section 5 with architectures, parameter
  counts, optimizer/loss explanations, fixed-epoch policy, and synthetic checks.
- Reuse/adaptation: inherit the supplied SparseMLP forward pass for one hidden
  layer and extend its output with a second affine/ReLU/dropout hidden stage.
  Adapt SAGA to precomputed CSR, retain sparse input minibatches, and avoid the
  supplied trainer's outer-held-out checkpoint selection.
- Baselines/metrics: implement training priors, hypothesis-block ablation with
  shared training preprocessing, and the frozen class-order/NLL/F1 conventions.
- Persistence: save weights together with preprocessing/settings/class mapping;
  verify exact text-pair probability reloads and detect changed files.
- Checks: controlled sparse/dense forward equivalence, finite gradients,
  learning/seeded evaluation, disabled evaluation dropout, solver convergence
  flags, argmax ties, NLLLoss equivalence, all-nine-system bundle round trips,
  and hypothesis-only invariance to changed premise content.
- Scope: synthetic arbitrary red/green/blue targets only; two neural epochs are
  a software-check override. No SNLI classifier is trained and the experiment's
  epoch count remains unset for the forthcoming measured pilot.
- Verification: all 29 project tests pass. The notebook executes all 24 code
  cells in a fresh kernel with no error/widget outputs. All 11 synthetic model
  bundles have valid probabilities/finite NLL and zero reload difference; all
  four synthetic SAGA fits report convergence. No SNLI score is inferred.
- Student review to add: explain logits versus probabilities/decisions,
  LogSoftmax/NLLLoss, sparse first-layer multiplication, regularization and
  optimizer updates, online versus evaluation metrics, parameter counts,
  the fixed checkpoint policy, and the precise hypothesis-only ablation.

## 2026-11-05: Step 6 measured pilot and frozen computation budget

- Assistance: implement `snli_pilot.py`, a separate pilot policy and selected
  frozen configuration, complete notebook section 6, and update the plan,
  methods, findings and reproduction instructions.
- Data handling: select fixed complete premise groups, preserving source IDs
  and order; fit BPE/bigrams on 6,001 pilot training rows only; evaluate on
  1,202 independent eligible validation rows. No official test predictions or
  scores are produced. Pilot metrics are tuning evidence, separate from CV.
- Models: add a training observer for pilot validation curves and a temporary
  neural CPU-thread setting, with restoration. The observer does not change
  the fixed epoch loop; a test verifies identical weights to an unobserved fit.
  Run nine primary designs plus class-prior and hypothesis-only baselines.
- Selection: compare candidate neural epochs `[3,5,8,12]` by the predeclared
  earliest-near-best mean-accuracy rule, selecting 3. Retry nonconverged SAGA
  fits at a larger iteration limit; record the initial 300-iteration failures
  and freeze ceiling 1,000 after all final pilot logistic fits converge.
- Computation: save preprocessing/fit times, sparse sizes and sampled process
  memory; use explicit linear projections with a factor-of-two allowance to
  select the 60,000-row target. Actual whole-group pool size is 60,005. Freeze
  its 15 checked grouped fold pairs separately from the original full protocol.
- Persistence: retain candidates/curves, metrics, original-ID predictions and
  complete model bundles; verify normalized probabilities, NLL agreement and
  exact raw-text reloads. Check saved artifacts/configuration by checksum;
  notebook inspection loads the report without rerunning the pilot.
- Verification: all 35 project tests pass; all 29 notebook code cells execute
  in a fresh kernel with no error/widget outputs. Inspect the saved learning
  curve figure. The actual pilot completes in 73.83 seconds; the approximately
  58-minute primary CV projection is a planning estimate, not a measured run.
- Student review to add: explain the independent pilot's role, group sampling,
  overfitting and NLL-versus-accuracy curves, fixed common epochs, solver
  convergence, sparse versus process memory, projection assumptions, and why
  scores from the 60,005-row CV pool must not be described as full-data SNLI CV.
  Check these choices before Step 7's reported comparisons.
