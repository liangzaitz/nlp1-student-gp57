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

## 2026-11-06: replace combined counts with bigram-only counts

- User decision: replace the third representation with pure bigram counts.
  Earlier entries describe the previous design and remain historical. Chat
  explanations covered fold-specific BPE, vocabulary versus feature inventory,
  indicators versus counts, sparse sentence-pair inputs and the depth comparison.
- Assistance: change the active configuration and encoder to `bigram_counts`,
  with layout `[P bigrams; H bigrams]` and dimension `2B`. Both sentence roles
  still share the fold's BPE vocabulary and training-only bigram inventory.
  Each model uses only the selected pair-count columns. An internal combined
  cache supports all three representations without repeated tokenization.
- Fit/interpretation: preserve frequency minimum 2, inventory cap 5,000,
  complete premise groups and protected holdouts. Explain zero bigram blocks
  for one-token inputs/unretained pairs, absence of unigram fallback, and equal
  10,000-coordinate widths/parameter counts at the fitted inventory sizes.
- Persistence: bump feature schema to `v2`, preserve old caches and combined
  pilot/frozen configuration, and create separate current
  `configs/snli_project1_frozen_bigram.json`. Reject loading a pilot report with
  a different representation set. Historical model bundles require earlier code.
- Refreshed evidence: execute the complete measured pilot for the revised
  nine-system matrix plus two baselines. Pilot `454dcdee4887d720` completes in
  55.22 seconds on 6,001 training / 1,202 validation pairs; probabilities and
  NLL checks pass, with zero raw-text probability difference after reload.
  Epoch selection still chooses 3; the common SAGA ceiling remains 1,000.
  Bigram-only SAGA converges in 115 iterations. The 60,005-row pool is unchanged,
  so its checked fold assignments are reused. The new primary runtime projection
  is about 42.5 minutes with the factor-of-two allowance, not measured CV time.
- Verification: all 36 project tests pass, including hand-computed pure-bigram
  values, repeated-pair counts, role-specific names, zero blocks with known
  unigrams but no retained pair, cache reconstruction and all-nine-model reloads.
  All 29 notebook code cells execute in a fresh kernel, rebuilding the full-fold
  features under a new cache ID and showing refreshed tables/curves/parameters.
  The generated learning-curve figure is inspected. No reported CV training or
  official test scoring is started.
- Student review to add: distinguish a BPE vocabulary, retained-token-pair
  inventory, internal cache and actual model input. Explain which information
  is traded when replacing unigram counts with bigram counts. Treat lower
  bigram-only pilot scores as preliminary observations, not a final CV ranking.

## 2026-11-06: Step 7 controlled cross-validation

- Assistance: implement `snli_experiments.py`, a resumable paired CV runner and
  read-only report loader, with the unchanged frozen bigram-only recipe.
  Extend notebook/documentation to explain the comparison and its artifacts.
- Training boundaries: fit preprocessing within each outer training fold;
  reuse it across the nine primary designs and two baselines. Pass only training
  matrices/labels into the model helper, with no outer-held-out observer. Keep
  three neural epochs, fixed model seed and SAGA settings. Test remains unscored.
- Persistence: save original-ID held-out predictions, class metrics/confusions,
  training histories, convergence flags/times/sizes and full prediction bundles
  for each fit. Check probability validity and full raw-text reloads. Aggregate
  only the exact complete paired grid; keep failures/partial work visible.
- Resume: append-only attempts, atomic state/complete markers, a process lock,
  immutable checked successes and separate source/config/protocol/version run
  identities. Preserve incomplete feature writes before retrying; reject
  corruption of completed artifacts. Notebook report reading performs no CV fits.
- Verification so far: all 40 project tests pass. New checks exercise a bounded
  run, simulated fit failure, resumed completion without changing earlier
  successes, exact held-out coverage/pairing, training-only labels, prediction
  corruption, missing grid entries, completed-run reuse and CPU-thread restoration.
- Reported execution: run `713ab1f5b3cb5801` completes all 135 primary and 30
  baseline/diagnostic fits. All probabilities/ID checks and full held-out raw-text
  reloads pass; the official test remains unscored. Saved outputs include
  1,980,165 predictions, paired accuracies, class reports, convergence,
  runtime/size records and standalone fold-score/training-curve figures.
  Recorded execution is 21.62 minutes; 0 logistic fits report nonconvergence.
  Interpret means/SDs descriptively; Bayesian posterior conclusions remain Step 8.
- Notebook verification: all 34 code cells execute in a fresh kernel with no
  error/widget outputs. Section 7 reads checked saved reports, validates the
  paired grid/shared per-fold features, displays metrics/curves/convergence and
  inspects the prediction format. Both standalone figures have been inspected.
  The plan/findings/method tracker and reproduction instructions record the
  measured results. No final model has been selected or evaluated on test.
- Student review to add: distinguish split seeds from model seeds, pipeline
  variation from initialization variation, online training curves from held-out
  scores, probability NLL from argmax accuracy, and descriptive fold SDs from
  independent uncertainty estimates. Step 8 will quantify paired differences.

## 2026-11-06: Step 8 Bayesian comparisons

- Assistance: implement `snli_analysis.py` and an independent analysis policy;
  extend the notebook, plan, findings, method tracker and reproduction commands.
  Reuse the supplied `BenavoliCorrelatedT` wrapper without changing training
  sources or the frozen recipe. No models retrained or official test scored.
- Inputs: checksum-verified CV run `713ab1f5b3cb5801`, 15 scores per system,
  identical `(repetition, seed, fold)` alignment and explicit `runs=3`. Compare all
  18 one-factor pairs plus 10 separate baseline/ablation pairs, not a set chosen
  by observed rankings. Preserve the primary accuracy ROPE ±0.005.
- Method: analytic Student-t region probabilities and 95% equal-tailed intervals,
  Nadeau–Bengio scale correction, correctly distinguish Student scale from actual
  SD, handle zero variance as a point mass. Record optional ROPE sensitivity
  with the original width clearly identified. Explain conditional assumptions,
  dependent folds, single-corpus/fixed-seed scope and lack of a joint ranking.
- Sources actually consulted: tutorial `analysis.py`, installed baycomp 1.0.3
  source, [baycomp API](https://baycomp.readthedocs.io/en/latest/classes.html),
  [implementation](https://baycomp.readthedocs.io/en/latest/_modules/baycomp/single.html),
  [Benavoli et al., Section 3](https://www.jmlr.org/papers/volume18/16-305/16-305.pdf),
  and the course's repeated-experiment discussion (HC2a PDF page 47).
- Execution: analysis `3b4f7bdb5c7c300b` saves 28 posterior rows/individual plots,
  420 paired differences, 84 sensitivity rows and an overview. Indicators versus
  unigrams is largely practically equivalent; pure bigrams lose about 7–8 pp.
  MLPs beat logistic beyond the ROPE; extra depth has less decisive practical
  benefit. The fixed hypothesis-only ablation is primarily inside the ROPE.
- Verification: all 47 project tests pass, with seven new tests covering paired
  keys/metadata, missing/duplicate scores, scale/interval/CDF agreement, runs=3,
  actual posterior SD, sign reversal, sensitivity, zero variance and report
  reuse/corruption/source changes without fitting. All 39 notebook code cells
  execute in a fresh kernel with no error/widget outputs. The overview and
  representative density plots have been inspected.
- Student review to add: explain B−A, fraction versus percentage points, ROPE
  versus error bars, posterior probability versus p-value, scale versus SD,
  common-correlation approximation and fixed-seed limitations. A mostly positive
  posterior can still imply practical equivalence. Step 9 error analysis/design
  selection remains next; no final design or official test result is asserted.

## 2026-11-06: Practical-style notebook and revised optimization priority

- User request: make our notebook akin to the supplied Practical 2 notebook.
  Read the reference through the GitHub connector (blob SHA
  `df0d7ee10b64e78d59946d92637cab97a9ecd378`) and inspect its cells: short
  explanations, visible computations, prepared inputs, model progression,
  training curves and interpretation. Source: [course practical](https://github.com/S-Chaudhuri/uva_nlp1_2025/blob/main/NLP1_2025_Practical_2_%28student%29.ipynb).
- Assistance: restructure `practical_1.ipynb` around SNLI examples, BPE vocabulary,
  role-specific sparse features, manual logistic logits, explicit neural first
  layer, loss/gradient updates, baseline results and Bayesian interpretation.
  Move implementation/environment checks and the old pilot to appendices.
  Expose a 30-update/16-training-example teaching trace, clearly separate from
  reported experiments. Preserve reusable modules and saved baseline artifacts.
- Scope: use the reference's presentation style. Retain the local Project 1
  requirements and four-deck course scope; do not adopt the other notebook's
  dataset, architectures, test-scoring pattern or grading/report requirements.
- Revised priority: record the accepted tokenizer/model optimization plan before
  Step 9. Section 6 shows proposed ranges and full-validation/fold-local selection
  as pending. No search runner, optimized settings or new validation/CV/test
  results are claimed. The original baseline's three-epoch recipe remains an
  immutable historical result, not a permanent search restriction.
- Verification: all 46 code cells run in a fresh kernel without error/widget
  outputs. Notebook schema and Python code syntax pass; manual logistic/model
  probabilities and sparse neural forward passes agree, teaching losses are
  finite, existing cache/pairing/convergence/reload checks pass. Previously
  recorded 47 production tests remain applicable because training sources and
  frozen recipes are unchanged. Notebook HTML is generated for visual review.
- Documentation: update the README, execution plan, method tracker and findings.
  A local pre-refactor notebook snapshot is preserved under
  `results/project1/notebooks/practical_1_before_style_refactor.ipynb`.
- Student review: explain each input/output and computation; distinguish the
  tiny learning demonstration, historical baseline, proposed tuning experiment
  and one-time final test. New tuning and error analysis remain outstanding.

## 2026-11-06: Move notebook-used implementation into notebook cells

- User clarification: code used should be written in notebooks, like the supplied
  practical, rather than hidden behind local Python-helper imports.
- Assistance: put actual data/audit/fold/BPE/feature/model/training/persistence/
  CV/Bayesian definitions into executable cells before their calls. Preserve
  standard-library/third-party imports, make private function names distinct in
  the shared namespace, and remove all imports of local `part1` implementations.
  The notebook performs no dynamic loading/execution of `.py` files.
- Provenance: notebook-generated artifacts hash executable cell source, excluding
  outputs. Retain original CLI files, frozen recipes and immutable baseline
  reports. Instantiate saved weights using notebook-defined model classes.
  Recompute all 28 saved posteriors from the notebook's literal definitions.
- Verification: execute all 132 code cells in a fresh kernel with local helper
  imports blocked; verify notebook/Python structure and no undefined globals.
  Separately exercise notebook-only CV on temporary synthetic data: bounded
  2-fit start, resume to 44 complete fits, checked report reload, then generate
  and independently recompute 28 Bayesian contrasts. This caught and corrected
  a missing third-party StratifiedGroupKFold import on the new-fold branch.
- Execution boundaries: long-run CV/analysis switches default false. No reported
  SNLI CV retraining, tokenizer/model search or official test scoring is started.
  The new validation-driven optimization remains the next unimplemented task.
- Disclosure: implementation definitions are adapted from our existing project
  modules and supplied course BPE/SparseMLP/Bayesian helpers, rather than copied
  from the external notebook's different task. Keep the earlier module-based
  experiment provenance and explain the new notebook execution path accurately.
- A pre-change notebook is preserved under
  `results/project1/notebooks/practical_1_before_inline_implementation.ipynb`.


## 2026-11-06: Implement the joint tokenizer/model search in the notebook

- User authorization: “go for it,” following the agreed validation-driven search
  plan and the requirement that implementation code appear literally in cells.
- Assistance: replace proposed section 6 with candidate generation, train-only
  preprocessing/cache construction, sparse feature construction, validation
  checkpoint selection, per-trial bundles/probabilities/audits, staged refinement,
  full-training finalist promotion, seeded ranking and resumable execution.
- Save `configs/snli_tuning.json` before real training. Keep historical frozen
  configs and CV/Bayesian artifacts unchanged. Search all three representations
  and classifiers; the third representation remains pure bigrams.
- Normalize with a serialized BPE lowercasing normalizer, retain punctuation and
  negation, and use training-only bigram frequencies. Source IDs and whole
  premise groups preserve the previous eligibility policy.
- Budget: 216 coarse configurations, up to 27 local refinements, then 42 full-data
  fits. Screen on a fixed 100k-row group sample using all eligible validation;
  full-data finalists use all eligible training rows. Repeat leading neural
  recipes with model seeds 0/3/5. Do not use official test scores.
- Train neural candidates with validation accuracy/NLL checkpoint selection,
  patience 3 and initial/absolute epoch ceilings 30/60. Restore best weights.
  Exclude nonconverged logistic candidates. Save review flags for neural ceilings.
- Verify six focused notebook-source checks and fresh-kernel Run All of all 141
  code cells, with local helper imports blocked. Integration checks exercise a
  full temporary search, interruption/resume, promotion, multi-seed ranking,
  validation-only vocabulary exclusion, raw-text reload and corruption rejection.
- Start the real SNLI search through a detached execution notebook that copies
  literal definition cells. A launcher startup extraction bug was caught before
  real trials and fixed; the canonical teaching notebook passed independently.
  The worker records its PID/status, execution copy and log under ignored results.
- Update README, plan, method tracker, findings and configuration documentation.
  Clearly distinguish implemented selection, ongoing real execution, later tuned
  outer CV/Bayesian comparisons and eventual test evaluation. No claim that
  development validation leaders are optimized generalization estimates.

- Final verification: all 53 repository tests pass (47 existing plus six tuning
  tests). Refresh section 6 from the checked live report without changing source;
  four initial completed SNLI trials are visible with validation curves. Current
  initial development run was `efe452ba59175f2f`; it was subsequently retired
  for the scheduler correction below.

- Scheduler review found that repeating one balanced permutation coupled some
  learning-rate/dropout choices. Independently shuffle balanced repeated levels
  for each setting; add a coverage check that rejects this coupling. Interrupt
  the initial worker after nine completed trials, preserve those artifacts/logs
  in its original run directory, and start the corrected source-identified run.
  No completed artifacts are silently migrated between source identities.
- Rerun all six focused checks and the complete 141-cell notebook successfully
  after correction. The 47 unchanged module tests also pass (53 tests total).
  The report display uses bundle creation times for chronological plotting;
  source fingerprints exclude display/control cells and remain stable when
  refreshing progress. Tuned outer CV/Bayesian comparisons remain pending.

- Corrected active development run: `d79628ac9dfb0954`; its worker executes the complete
  saved policy automatically and the live report distinguishes incomplete search
  progress from final selected recipes.


## 2026-11-06: Reduce tuning to a 15-minute maximum

- User request: “can we do alot less ... like 15 min.” Stop the large worker;
  preserve its 11 completed observations and archive its execution/log/status.
- Replace the default tuning policy with 27 planned observations (three explicit
  joint recipes × nine systems), one seed, at most eight epochs and a 900-second
  wall-clock allowance including preprocessing. Preserve the old extensive
  policy separately and make its invocation explicit/optional.
- Retain the substantial 100,001-row training sample and all 9,839 eligible
  validation examples. Explicitly reuse nine compatible baseline observations;
  verify training/feature source signatures, data and exact rows, recipe/seed,
  stopping history and saved hashes. Original trial records remain immutable.
- Implement the new runner, compatibility checks, time-limit context and report
  reader literally in notebook section 6.9. Default search/report functions route
  to the short policy. No full-data promotion, refinement stage, multi-seed
  confirmation or automatic second allowance is scheduled.
- Verify eight focused checks, including real timer expiry/handler restoration,
  complete temporary 27-observation selection, reuse/resume, partial budget reports
  and refusal to reuse an unverifiable implementation. Start the bounded SNLI job.
- Update plan, README, findings, method tracker and config docs. Label selection
  as best tested on a subset/single seed; do not claim thorough optimization or
  new CV/test estimates from this smaller budget.

- Completed the bounded SNLI search: `4baddae6266ac867`, 27 observations, nine
  reused/18 new fits, 7.97 minutes including preprocessing/checks. Record
  best tested recipes and the remaining epoch-ceiling review flag; no follow-up
  training is queued. All 55 repository tests pass. The best validation accuracy
  is 67.83%; its 0.09 pp gain over the same-pool baseline is small and has worse
  NLL, so do not claim a robust generalization improvement.

- Final Run All verification: all 143 notebook code cells execute in a fresh
  kernel with local helper imports blocked; no errors/widgets. Save the complete
  checked report/figures and retain the default 900-second cap for future runs.


## 2026-11-06: Review and streamline the full report

- User request: check the report/code completeness, compare presentation with the
  linked NLP1 practical and reduce complexity. Read the project requirements,
  full notebook and notes, saved data/predictions/manifests, and the GitHub
  reference notebook (blob `df0d7ee10b64e78d59946d92637cab97a9ecd378`).
- Replace the 219-cell/3,211-code-line main notebook with 54 cells, 32 code cells,
  852 code lines: one encoder, one training function, short search/CV loops,
  bundle persistence and direct corrected-t computation. All custom code used
  remains literal notebook code; no project helper imports or file execution.
  The external reference informs presentation, not the task/model selection.
- Preserve the original detailed implementation in `practical_1_details.ipynb`,
  with outputs cleared and training flags false. Keep the pre-change executed
  notebook and long-form notes locally under ignored results. Existing experiment
  artifacts, frozen recipes and Python experiment modules are unchanged.
- Check all 165 saved CV fits and all 27 validation observations against row IDs,
  gold labels, normalized probabilities, metrics and hashes. Recompute all 28
  Bayesian comparisons and 84 sensitivity rows; replay 11 systems from raw text.
  No numerical discrepancy at the documented tolerances. Inspect output figures.
- Add eight focused tests of the new literal code. Check direct feature counts,
  empty/boundary cases, training agreement with the historical implementation,
  best-checkpoint restoration, raw-text bundles, corruption detection, search
  resume, tiny complete CV and actual timer expiry. Retain historical notebook
  runner tests against the detailed archive. All 63 repository tests pass.
- Execute all 32 main code cells in a fresh kernel with project-helper imports
  blocked; no cell errors or widget views. Only the small teaching example and
  synthetic tests train new models; no new SNLI sweep or test predictions.
- Rewrite current README/plan/findings and refresh the course tracker. Correct
  stale pending-tuning text, separate validation selection from baseline CV,
  state the small 0.09 pp gain/worse NLL and visible NLL overfitting, and retain
  the active epoch-ceiling limitation. Error analysis, final selection/refit/test
  and student oral preparation remain unfinished; the report does not claim them.


## 2026-11-06: Restore visible ROPE sensitivity results

- User asked what happened to ROPE sensitivity after simplification. The numerical
  checks for all 84 rows were retained, but the displayed cell had shown only
  context comparisons at the primary threshold. Restore a compact table of all
  18 primary contrasts across ±0.25/±0.5/±1 pp, computed directly in the notebook.
- Explain the unchanged posterior versus changing practical region, the close
  depth comparison's dependence on tolerance, rounding near 100%, and the
  unchanged preselected primary ±0.5 pp. No training or threshold selection.


## 2026-11-06: Step 9 — held-out error review and final recipe

- User request: proceed to the next step. Use saved baseline predictions only;
  no new SNLI model training or official test evaluation.
- Add notebook-contained OOF alignment, class reports/confusion figures, lexical
  overlap/negation slices and shared-error counts. Preselect split seed 0 so each
  of 60,005 examples contributes one prediction per system, rather than treating
  repeated CV predictions as independent examples.
- Select 50 distinct examples/premise groups deterministically, covering directed
  confusions, confident shared-neural errors and bigram/H-only disagreements.
  Read every selected premise/hypothesis with its gold and comparative predictions;
  write literal AI-assisted categories/notes. Display illustrative cases and export
  full texts/notes and a source/hash manifest under ignored results.
- Flag 11 wording/label uncertainties without changing any gold label or metric
  denominator. Explain purposive selection, confounded surface slices, and limits
  of causal and noise-prevalence claims. These are assistant-authored first-pass
  annotations; students must verify them and understand the examples.
- Freeze a separate `configs/snli_final.json`: 5k BPE, unigram counts, MLP1/128,
  seed 0, five epochs from its best validation-accuracy checkpoint. Explain the
  small larger-vocabulary accuracy gain versus worse NLL/more parameters, the
  accuracy/NLL/depth trade-off, and transferring epochs to the larger final pool.
  Historical three-epoch CV and all old experiment artifacts stay unchanged.
- Add three meaningful synthetic checks for OOF alignment/corruption, slice/shared
  denominators and deterministic premise-disjoint case selection. Execute the
  notebook with project-helper imports blocked; all 38 cells execute without errors.
- Update the plan, findings, README, course tracker and configuration notes.
  Step 10 final fitting/test and student oral preparation remain pending.

- Final verification: all 66 repository tests passed; all 38 main code cells
  executed in a fresh kernel with helper imports blocked and no error/widget
  outputs. Save the confusion figure as a standalone PNG with an artifact hash.


## 2026-11-07: Step 10 — frozen final fit/test and graphs

- User request: implement Step 10 and provide graphs. Retain all custom code in
  literal main-notebook cells; do not import project implementation modules.
- Implement final train+validation eligibility excluding only test premises,
  recovering 72 train rows excluded by the development overlap policy. Verify
  all 559,164 rows, source IDs and fixed recipe before fitting fresh BPE/weights.
- Use the unchanged frozen five-epoch MLP1 recipe, seed 0, CPU/one thread and
  900-second training allowance including preparation and bundle reload checks.
  Complete in 2.98 minutes. Save exact training IDs, history, source notebook,
  versions and a complete raw-text prediction bundle.
- Make one final-model probability pass on 9,824 cleaned official test examples.
  Test accuracy 74.58%, macro-F1 0.7451, NLL 0.6886.
  Save immutable predictions, class report/confusion and file hashes. Reruns read
  and verify saved probabilities; a lock and test-start journal prevent automatic
  duplicate scoring. No model/epoch/threshold/calibration is chosen from test.
- Add four exported graph sets: online learning curves, test confusion matrices,
  per-class precision/recall/F1 and confidence/reliability. Ten fixed bins give
  descriptive ECE 0.0728; these diagnostics do not fit a calibration transform.
- Add three meaningful tests for recovered validation overlap and train-only BPE,
  one-pass reuse/corruption detection, budget/interruption guards and confidence
  bin boundaries. All 69 repository tests pass. All 42 main code cells execute
  with local helper imports blocked; RUN_FINAL returns to false in the saved notebook.
- Update the plan/findings/README/method map/configuration notes. The frozen
  selection JSON remains unchanged and retains its pre-test history. Student
  code/error-note review and oral preparation are still required in Step 11.
