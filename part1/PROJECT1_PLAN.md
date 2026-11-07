# Project 1: fold-trained BPE and vocabulary-size selection

The active experiment is entirely in [practical_1.ipynb](practical_1.ipynb).
Tune **only BPE vocabulary size: 1,000 / 2,000 / 3,000**. Keep all other
hyperparameters fixed. Tables stay in notebook outputs; plots are exported as
PNG and PDF. There are no trial logs or checkpoints.

The fresh run is **complete**: 15 CV tokenizers, 135 classifier fits, automatic
selection of 2k / indicators / MLP2, final refit and selected-model test. Mean CV
accuracy is **62.69%**, test accuracy **74.48%**. Nine figure sets and all tables
are saved. Student code/error review and oral preparation remain.

## Experimental flow

1. Load the pinned, cleaned SNLI splits. Combine training + validation and exclude
   exact premises shared with test: **559,164 development pairs**.
2. Keep the same **60,002-row** complete-premise-group subset, sampling seed 0.
   Use the same five stratified grouped folds, split seed 0, for every candidate.
3. For each fold and each vocabulary size, train a **new BPE on fold-training
   premises and hypotheses only**. Fit its bigram inventory on those same rows.
   Transform validation with these fixed mappings; never fit on validation text.
4. Share that tokenizer/inventory across indicators, unigram counts and pure
   bigram counts, each with logistic, MLP1 and MLP2: **15 CV tokenizers, 135 fits**.
5. Show all 27 candidate summaries, the best system per vocabulary, and an
   accuracy matrix/deltas versus 2k for each fixed representation/classifier.
   Report dimensions, parameters, fit time and tokenization length too.
6. Select highest mean CV accuracy; lower NLL breaks an exact tie. Remaining exact
   ties use smaller vocabulary, then lexical representation/classifier order.
   Show paired Bayesian comparisons and ROPE sensitivity as descriptive evidence.
7. Inspect selected-system OOF errors. Train a **fresh tokenizer of the selected
   size** and a fresh selected classifier on all development rows. Relearn the
   final bigram inventory there. CV model weights are not transferred.
8. Score **only that final selected pipeline** on official test. Display its
   metrics, per-class scores, errors and diagnostic graphs.
9. Review methods, results and errors as a group; prepare the oral assessment.

## Hyperparameters held fixed

| Setting | Value |
| --- | --- |
| Dataset revision | `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b` |
| BPE tuning candidates | **1,000 / 2,000 / 3,000** |
| Other BPE settings | Minimum frequency 2; `[UNK]`; Metaspace; preserved case/punctuation |
| Bigram features | Frequency ≥2; **cap 2,000 at every BPE size**; no unigram fallback |
| Pair features | Separate premise/hypothesis blocks; binary presence or raw counts; sparse float32 |
| Logistic | L2, SAGA, C=1, maximum 1,000 iterations, tolerance 0.0001 |
| Hidden units | `[128]` or `[128,64]` |
| Activation/dropout | ReLU / 0.1 |
| Neural optimizer | AdamW, learning rate 0.001, weight decay 0.0001 |
| Batch/epochs | 128 / **3 in CV and final fitting**; no early stopping |
| Sampling/split/model seeds | 0 / 0 / 0 |
| Compute | CPU, one PyTorch thread; same five folds throughout |
| Selection metric | Mean CV accuracy, then lower NLL for an exact tie |
| Supporting metrics | Natural-log NLL, macro-F1, descriptive fold SD |
| ROPE | ±0.005 accuracy; sensitivity ±0.0025 and ±0.01 |

Indicator/unigram inputs have up to `2V` columns; pure bigrams have up to 4,000
because the inventory cap remains fixed. Vocabulary size therefore changes
segmentation and, for unigram inputs, model size. Matching dimensions across all
three vocabulary candidates would also change the bigram cap, contrary to the
single-hyperparameter design.

## Reporting and limitations

Use paired vocabulary curves within each fixed system. A comparison of per-size
winners may also change architecture/representation and is not a vocabulary-only
contrast. Bayesian tables compare systems at the selected size and all vocabulary
pairs for the selected system. These are post-selection descriptions using the
same folds, not independent confirmation or global ranking probabilities.

The best CV mean is a selection score among 27 candidates. It is not a nested-CV
estimate of the complete selection procedure. Five correlated fold measurements
provide limited posterior information. Final training uses much more labelled
data than each CV fit, so rankings need not transfer perfectly to the full pool.

Earlier work already inspected official test results. Disclose that history and
keep current test scores out of all new selection decisions. Historical shared-BPE
results and figures remain references; never substitute their metrics here.
See [findings](PROJECT1_FINDINGS.md) and [assistance](AI_ASSISTANCE.md).
