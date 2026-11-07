# Project 1 plan: SNLI

Follow [the assignment](project.ipynb) and the four supplied lecture decks, mapped
in [COURSE_METHODS.md](COURSE_METHODS.md). The main deliverable is
[practical_1.ipynb](practical_1.ipynb), with all its custom implementation in cells.
The experimental scope is indicators / unigram counts / **pure bigram counts**,
combined with logistic / one-hidden-layer MLP / two-hidden-layer MLP.

## Completed

- [x] **1. Setup:** environment, seeds, pinned dataset revision and assistance/source records.
- [x] **2. Data:** download, clean, preserve IDs, inspect labels/lengths/overlap.
- [x] **3. Evaluation:** holdout-disjoint pools, premise-grouped folds, accuracy/NLL/F1,
  preselected ROPE ±0.005 accuracy and test reservation.
- [x] **4. Features:** training-only BPE and bigram inventory, separate premise/hypothesis
  blocks, sparse indicators/counts/pure bigrams; no unigram fallback for bigrams.
- [x] **5. Models:** logistic, one/two-hidden-layer MLPs, class-prior and hypothesis-only
  diagnostics, probability scoring and complete raw-text prediction bundles.
- [x] **6. Baseline schedule:** historical small engineering pilot; freeze three epochs,
  5k BPE, solver ceiling and a 60,005-row complete-group CV pool.
- [x] **7. Baseline CV:** five folds × three split seeds × 11 systems = 165 fits.
  Neural checkpoints use the fixed schedule, never outer held-out selection.
- [x] **8. Bayesian analysis:** 28 paired correlated-t comparisons, 18 primary one-factor
  contrasts, ROPE sensitivity and conditional pairwise interpretation.
- [x] **Short tuning extension:** three joint recipes × nine systems, 27 observations,
  100,001 training / 9,839 validation rows, one model seed, up to eight epochs,
  patience three. Nine compatible trials reused, 18 new fits, 7m 58s.
- [x] **Report review:** simplify the main notebook; independently check saved metrics,
  Bayesian results and raw-text reloads; distinguish baseline CV from tuning evidence.

These meet the core implemented representation, architecture, CV and Bayesian
comparison requirements. The nine-system grid, three CV repetitions, diagnostics
and final test evaluation are design choices beyond the minimum checklist.

## Completed: Step 9 — errors and final design

- [x] Review 50 distinct examples/premise groups, covering all six directed anchor
  confusions, confident shared failures and disagreements with bigram/H-only models.
  Notes are AI-assisted; student verification remains part of the group review.
- [x] Use one complete out-of-fold repetition (split seed 0, 60,005 rows), with
  class confusion matrices, per-class scores, lexical-overlap/negation slices
  and paired shared-error counts. No repeated-prediction pseudo-replication.
- [x] Discuss paraphrase, unsupported detail, attribute/action conflicts, predicate
  roles, commonsense and contextual negation. Flag 11 text/label uncertainties
  without changing the gold labels. The purposive sample is not a noise estimate.
- [x] Choose 5k BPE / unigram counts / 128-unit one-hidden-layer MLP, balancing
  accuracy, NLL and size. The larger vocabulary's same-pool gain is only 0.09 pp,
  with worse NLL and about 60% more parameters.
- [x] Document the compact unigram MLP2's active ceiling; do not extend training.
- [x] Freeze [snli_final.json](configs/snli_final.json): seed 0, five fixed epochs,
  taken from the chosen recipe's best validation-accuracy checkpoint. Explicitly
  record transfer of that schedule from 100,001 rows to the larger final pool as
  an unconfirmed choice. Historical three-epoch CV scores remain unchanged.
- [x] Save complete notes, OOF metrics, confusions, slices and a source/hash manifest
  under `results/project1/error_analysis/713ab1f5b3cb5801/split_seed_0/`.

A new tuned-CV claim would need its own affordable selection/evaluation design,
with selection inside outer training folds. It is an optional extension, not a
reason to block error analysis or repeat the large stopped search. The current
short search gives validation-selection evidence, not nested-CV results.

## Completed: Step 10 — final refit and test

- [x] Fit fresh train+validation BPE and the frozen MLP on all **559,164** eligible
  rows. Exclude test premises, retain train/validation overlaps and use five epochs.
- [x] Finish preparation, fitting and save/reload checks in **2.98 minutes**,
  within the fixed 900-second allowance. No test-based stopping or model changes.
- [x] Save the complete pipeline, exact training IDs, history, package/source hashes
  and a source-notebook snapshot in `results/project1/final/snli_5k_unigram_mlp1/`.
- [x] Generate one final-model probability set on all **9,824** cleaned test rows:
  accuracy **74.58%**, macro-F1 **0.7451**, NLL **0.6886**.
  Preserve class reports, confusions and predictions. Run All reuses these artifacts.
- [x] Export learning curves, test confusion matrices, per-class metrics and
  confidence/reliability plots. Fixed-bin confidence analysis is diagnostic only.
- [x] Test full-pool eligibility, successful reload/reuse, budget failure and the
  one-pass guard; leave the original frozen selection configuration unchanged.

Official test was scored only after the final recipe/schedule were frozen.
No model, threshold or calibration was fitted to its results.

## Next: Step 11 — package and prepare

- [ ] Review all code/explanations as a group and resolve remaining student questions.
- [x] Repeat fresh-kernel Run All after final changes; keep sources, dependency versions,
  configurations, results and assistance disclosure reproducible.
- [ ] Prepare the repository URL and up to six optional oral backup slides covering
  the assignment questions. Retain clear notes for the later blog.

**Compute constraint:** approximately 15 minutes per new training invocation;
training flags default to false. No large sweep, seed confirmation, full-data
search or follow-up worker is queued. The historical baseline took 21.62 minutes;
reproducing its entire CV grid needs separate planning under this constraint.

Use [PROJECT1_FINDINGS.md](PROJECT1_FINDINGS.md) for current evidence and limitations.
The detailed historical implementation remains in
[practical_1_details.ipynb](practical_1_details.ipynb); previous long-form notes are
preserved locally under `results/project1/review/before_streamlining/`.
