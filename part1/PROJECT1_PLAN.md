# Project 1: fixed-setting SNLI experiment

The active experiment is entirely in [practical_1.ipynb](practical_1.ipynb).
There is no hyperparameter search, separate validation-selection stage or result
logging. Saved notebook outputs contain the result tables and graphs.

The fresh 2k experiment is **complete**: 45 CV fits, automatic indicator-MLP2
selection, full refit and official test, with seven exported figure sets.
Mean CV accuracy is **62.60%**; test accuracy is **74.48%**.
Student code/error review and oral preparation remain.

## One experimental flow

1. Clean the pinned SNLI splits; combine training + validation and exclude exact
   premises shared with test. This gives **559,164 development pairs**.
2. Train one **2,000-token BPE** on both sentence roles of that development pool,
   without labels or test text. Freeze it for CV, final fitting and test encoding.
3. Select **60,002 rows** as complete premise groups, targeting 60,000 with seed 0.
   Use one stratified five-fold group split, seed 0, for all nine systems.
4. Compare indicators, unigram counts and pure bigram counts, each with logistic,
   one-hidden-layer MLP and two-hidden-layer MLP. Learn bigram inventories and
   classifier weights from fold-training rows only. Keep all training settings fixed.
5. Display mean CV accuracy, fold SD, NLL and macro-F1 for every candidate.
   Select highest mean accuracy, then lower mean NLL for an exact accuracy tie.
6. Show paired Bayesian correlated-t comparisons and ROPE sensitivity. These
   describe uncertainty and do not replace the stated model-selection rule.
7. Inspect one repetition of OOF errors. Refit the selected classifier on all
   development labels using the same shared BPE and fixed training schedule.
   Relearn the final bigram inventory from the development pool if needed.
8. Evaluate the final system on official test; show metrics, per-class scores,
   confusion matrices, online training curves and a compact error table.
   Add the tutorial-style fold boxplots with points, posterior/ROPE curves and
   a ROPE sensitivity heatmap; export plots as PNG and PDF.
9. Review code, scientific conclusions and errors as a group; prepare the oral.

## Frozen settings

| Setting | Value |
| --- | --- |
| Dataset revision | `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b` |
| BPE | Vocabulary 2,000; minimum frequency 2; `[UNK]`; Metaspace; preserved case/punctuation |
| Bigram features | Training-only inventory; frequency ≥2; maximum 2,000 types; no unigram fallback |
| Pair features | Separate premise/hypothesis blocks; raw counts or binary presence; sparse float32 |
| Logistic | L2, SAGA, C=1, at most 1,000 iterations, tolerance 0.0001 |
| MLP hidden units | `[128]` or `[128, 64]` |
| Activation/dropout | ReLU / 0.1 |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Batch / epochs | 128 / **3 in CV and final fitting**, no early stopping |
| Sampling / split / model seeds | 0 / 0 / 0 |
| Compute | CPU, one PyTorch thread; 45 CV fits |
| Primary / supporting metrics | Accuracy / natural-log NLL and macro-F1 |
| ROPE | ±0.005 accuracy; sensitivity ±0.0025 and ±0.01 |

The single five-fold repetition keeps the fresh experiment affordable relative to
our earlier approximately 15-minute target. Fewer paired scores mean greater
Bayesian uncertainty than in the former three-repetition design.

## Interpretation boundary

The shared BPE has seen held-out CV text, without its labels. These scores are
conditional on the shared tokenizer; they are not CV estimates of a pipeline
whose preprocessing was entirely fitted inside each fold. Test text never
enters learned preprocessing. Test-premise membership is used only to exclude
overlapping development groups.

Earlier work already evaluated a different model on test and explored tuning.
Those are historical facts, not stages of the current flow. This new experiment
must not be described as a never-before-seen first test or an isolated estimate of
the effect of vocabulary size. Historical configurations/results remain available
for reference but are not loaded by the active notebook. See
[findings](PROJECT1_FINDINGS.md) and [assistance](AI_ASSISTANCE.md).
