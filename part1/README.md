# Project 1: SNLI

Open [practical_1.ipynb](practical_1.ipynb). All custom implementation and result
tables are in notebook cells. It does not import project helper modules or read
external experiment configurations/results.

**Grouped CV → fold-trained BPE at 1k/2k/3k → select one candidate → full refit → test.**

Vocabulary size is the only tuned hyperparameter. At every size, compare
indicators, unigram counts and pure bigram counts with logistic regression,
one-hidden-layer MLP and two-hidden-layer MLP. Keep the bigram cap at 2,000 and
all classifier settings fixed, including three neural epochs in CV and final fitting.

## Run

Choose `.venv/bin/python` as the notebook kernel, or launch:

```bash
source .venv/bin/activate
jupyter lab part1/practical_1.ipynb
```

Run All performs **fresh fitting**: 15 CV tokenizers, 135 classifiers, then one
fresh final tokenizer/classifier. Save the executed notebook to retain the tables.
No result CSVs, trial logs or model checkpoints are written. PNG/PDF plots go to
[figures/project1_vocab_cv/](figures/project1_vocab_cv/). Existing cleaned SNLI is
reused; a fresh checkout downloads the pinned revision through Hugging Face.

Use Python 3.12 and the root `requirements.txt`, or recreate the recorded CPU
PyTorch environment with `environment.txt` and its stated package index.

## Read the report

| Section | Content |
| --- | --- |
| 1–2 | Fixed settings, eligible development pool and shared grouped folds |
| 3–4 | Training-only BPE/features and explicit fixed-epoch classifier fitting |
| 5 | Best per vocabulary, all 27 candidates, controlled size comparisons, dimensions/cost and vocabulary curves |
| 6 | Paired Bayesian system/size comparisons and ROPE sensitivity |
| 7 | One OOF repetition for selected-system errors |
| 8 | New selected-size BPE, full-data refit and selected-model-only official-test results |
| 9 | Tutorial-style fold boxplots/points, posterior/ROPE curves and sensitivity heatmap |

Development contains **559,164** training + validation pairs after excluding exact
test-premise overlaps. CV uses the unchanged **60,002-row** complete-group sample,
five folds and seed 0. Each fold/size fits BPE and its inventory only on training
text, and shares them across its nine classifiers. A fold's models and validation
rows use the same coordinates; other folds may learn different coordinates.

Choose the highest unrounded mean CV accuracy, lower NLL for an exact tie, then
smaller vocabulary and lexical system order. All three sizes use the same folds.
The selected CV maximum and subsequent Bayesian comparisons are post-selection
evidence, not an unbiased nested-CV performance estimate. Official test scores
are generated only for the selected final pipeline. Earlier test use is disclosed.

Historical `figures/project1_2k/`, Python modules, JSON configurations and the
`practical_1_details.ipynb` archive are not used by the active experiment.

## Completed results

| BPE size | Best tested CV system | Mean CV accuracy |
| --- | --- | ---: |
| 1,000 | Unigram counts / MLP2 | 62.27% |
| **2,000** | **Indicators / MLP2** | **62.69%** |
| 3,000 | Indicators / MLP2 | 62.55% |

The fixed rule selects 2k. Differences are small; see the matched-system curves
and posterior probabilities. Full refitting reproduces **74.48% test accuracy**,
NLL **0.6210**, macro-F1 **0.7439**. It selects the same final recipe as before.
CV took **10.09 minutes**, final preprocessing/fitting **60.25 seconds**.

All nine figure sets are inline and exported as PNG/PDF. Start with
[vocabulary effects](figures/project1_vocab_cv/vocabulary_effect.png),
[vocabulary posteriors](figures/project1_vocab_cv/vocabulary_posteriors.png) and
[fold scores](figures/project1_vocab_cv/cv_boxplots_and_points.png).

## Verification and sources

```bash
.venv/bin/python -m unittest discover -s part1/tests -v
```

Notebook tests check training-only BPE membership for every fold/size, tokenizer
reuse across systems, held-out coverage, feature counts/boundaries, fixed epochs,
selection and paired posterior calculations. Other tests cover retained modules.
See [requirements](project.ipynb), [plan](PROJECT1_PLAN.md),
[findings](PROJECT1_FINDINGS.md), [course methods](COURSE_METHODS.md) and
[AI assistance](AI_ASSISTANCE.md).
