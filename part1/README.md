# Project 1: SNLI

The current implementation covers workspace setup (Step 1), pinned SNLI
loading/inspection (Step 2), evaluation pools/saved grouped folds (Step 3),
training-only BPE/sparse sentence-pair features (Step 4), classifiers/baselines
(Step 5), and the measured SNLI pilot/frozen comparison budget (Step 6).
Reported CV comparisons are next in the
[execution plan](PROJECT1_PLAN.md). The [method tracker](COURSE_METHODS.md)
connects our choices to the supplied lectures and tutorial.

## Run the current implementation

From the repository root:

```bash
source .venv/bin/activate
python -m part1.project_setup
python -m part1.snli_data
python -m part1.snli_evaluation
python -m part1.snli_features --seed 0 --fold 0
python -m part1.snli_models
python -m part1.snli_pilot
python -m part1.snli_evaluation --config part1/configs/snli_project1_frozen.json
jupyter lab part1/practical_1.ipynb
```

In VS Code, select `.venv/bin/python` as both the interpreter and notebook
kernel. The notebook works when its working directory is the repository root
or `part1`; it finds the repository before importing the shared setup helper.
Use Run All to execute the implemented sections. The first Step 2 run downloads
SNLI; later runs reuse the local cache. Section 4 fits/reloads preprocessing for
one full outer fold. Section 5 checks the models using synthetic data only.
Section 6 loads the saved measured SNLI pilot and verifies its frozen comparison
folds. Run All does not retrain the pilot. Sections 7–11 remain pending;
reported CV model comparisons and final test scoring have not run.

Cached data is still the downloaded Hugging Face dataset. A message mentioning
offline mode means the loader used local files without contacting the Hub;
it does not mean SNLI is missing. Section 2.2 displays the split sizes and an
actual training example to confirm loading. Notebook progress bars are disabled
because their saved widget output may not render when the notebook is reopened.
The cache's `.lock` files coordinate access; the dataset rows live in Arrow files.

The data command and notebook use the same `snli_data.py` implementation. It
verifies the label mapping, filters invalid labels/unusable text, preserves
source row IDs and official splits, saves the cleaned DatasetDict, and produces
an exact duplicate/overlap audit, summary CSV files, and two figures. See
`PROJECT1_FINDINGS.md` for observed counts and the overlap policy in Step 3.

Step 3 reads the saved cleaned data without a Hub request. CV excludes training
premises present in either official holdout; pilot validation excludes test
premises. Final refitting is reserved for original train + validation excluding
test premises. These are separate selections; the cleaned official splits stay
unchanged. Only test premises/source IDs are used for these structural checks.

Five `StratifiedGroupKFold` folds for each seed `[0, 3, 5]` are saved under
`results/project1/folds/<protocol_id>/`. Generating full-SNLI folds takes longer
than reloading them. Repeated runs verify and reuse the assignments; model
hyperparameter changes do not regenerate them. A different data pool, split
configuration, evaluation policy, or splitter version gets a separate ID.
Step 6 saves a separate 60,005-row comparison protocol after selecting a
complete-premise-group subset for the computation budget.

Step 4 reuses the tutorial's BPE helper, with target vocabulary size 5,000,
minimum merge frequency 2, Metaspace `▁`, and no case/punctuation normalization.
It fits only the requested outer training partition, then learns a capped
inventory of within-sentence BPE bigrams from that same text. Premise and
hypothesis occupy separate blocks. Indicators, raw unigram counts, and
unigram-plus-bigram counts share tokenization/inventory and use float32 CSR.

`python -m part1.snli_features --seed 0 --fold 0` materializes the representative
full fold verified in notebook section 4 (439,403 train / 109,850 held-out rows).
Use another saved seed/fold to prepare that fold's own preprocessing. The other
14 full-reference folds are deferred. This feature command launches no
classifier training. It does not tokenize official validation/test text.
Reruns verify/reuse the feature cache; changed feature settings or code receive
a separate cache ID, while model-only changes retain the features.

For arbitrary sentence pairs, reload `PairFeatureEncoder` from the feature
directory and call `transform(premises, hypotheses, representation)`. It applies
the fixed mapping; empty/whitespace-only inference sentences get zero role
blocks, and unseen characters use `[UNK]`. Non-string inputs are rejected.

Step 5 adapts the supplied SAGA logistic approach to existing CSR matrices,
reuses the tutorial's sparse first hidden layer, and adds a two-hidden-layer
MLP. `fit_classifier(kind, X, y, config)` accepts training labels only. Neural
fits require a positive fixed epoch count; the reference/pilot configuration
keeps it unset, while Step 6's frozen configuration sets it to 3. Prediction
disables dropout. Histories record online training loss/accuracy, and logistic
results record convergence status.

`python -m part1.snli_models` fits/checks all nine feature/model combinations,
a training-class-prior baseline, and the configured hypothesis-only ablation
on tiny synthetic text with arbitrary software-check targets. Its two neural
epochs are a mechanical override, not a SNLI training choice. Each invocation
writes a new `checkpoints/implementation_smoke/run_*/` directory with checks and
complete prediction bundles. It reads no SNLI examples and reports no SNLI scores.

Use `save_prediction_bundle(...)` to save weights together with the correct
tokenizer/inventory, representation, input mode, class mapping, settings, and
fit information. `load_prediction_bundle(path).predict_proba(premises,
hypotheses)` reloads the complete text-pair pipeline; `.predict(...)` applies
the shared argmax rule. Bundle directories are not overwritten; checksums
detect changed model/preprocessing artifacts. Recreate the recorded environment
for reloads, especially the sklearn version used by joblib logistic estimators.

`score_probabilities` implements accuracy/NLL/macro-F1 and class/confusion
reports under the fixed Step 3 conventions. The hypothesis-only diagnostic
drops premise coordinates while retaining shared training preprocessing; its
tokenizer is not claimed to have been trained on hypotheses alone.

Step 6's pilot uses 6,001 training pairs and 1,202 official-validation pairs,
selected by complete premise groups with seed 0. It fits fresh training-only
preprocessing, trains all nine designs and two baselines, and saves curves,
timings, memory measurements, class reports, source-ID predictions and complete
prediction bundles. All 11 bundles reproduce probabilities exactly after reload.
Pilot validation is used for tuning; these scores are not reported CV estimates.

The predeclared epoch rule selects 3 neural epochs. Initial 300-iteration SAGA
attempts do not converge; retries converge under a 1,000-iteration ceiling.
The frozen recipe uses that ceiling and one neural CPU thread. Approximate
linear planning with a factor-of-two allowance chooses the 60,000-row target
(60,005 actual rows / 16,590 premise groups), estimating about 58 minutes for
135 primary fits plus fold preprocessing. Diagnostics, baselines and final
refitting add work; real runtime can differ. See [findings](PROJECT1_FINDINGS.md).

Use [configs/snli_project1_frozen.json](configs/snli_project1_frozen.json) for
Step 7, with protocol `3e3b8a39db826f1c`. The original full-reference protocol
`0dc6821564b1d3f0` and its feature cache are retained. Each new comparison
training fold must fit its own preprocessing. To prepare a comparison fold:

```bash
python -m part1.snli_features --config part1/configs/snli_project1_frozen.json --seed 0 --fold 0
```

The pilot command verifies/reuses completed artifacts on reruns. To inspect
without training, use `load_pilot_report()` from `part1.snli_pilot`, as notebook
section 6 does. Checksums detect changed artifacts/frozen configuration. The
controlled CV runner is the next implementation step.

Run the offline data, leakage, feature-arithmetic, and persistence checks:

```bash
python -m unittest discover -s part1/tests -v
```

## Recreate the recorded environment

The course README uses Python 3.14. Our currently verified environment is
**Python 3.12.3 on Linux x86_64 with CPU PyTorch**. `environment.txt` records its
installed package versions; it is a platform-specific snapshot, not a guarantee
of compatibility with Python 3.14 or a GPU installation.

To reproduce this environment on a compatible machine, create a Python 3.12
virtual environment, activate it, then install the snapshot:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r part1/environment.txt
python -m pip check
```

Alternatively, follow the root README/course setup and install
`requirements.txt`. Record and verify that environment separately instead of
assuming the supplied snapshot describes it.

## Configuration and outputs

[configs/snli_project1.json](configs/snli_project1.json) is the reference/pilot
configuration used by notebook sections 1–5 and the default setup command. It
records the pinned SNLI revision, seeds, initial feature/model settings, metrics
and ROPE; its epoch count stays unset. [configs/snli_pilot.json](configs/snli_pilot.json)
records the pilot sampling, search and runtime policy.
[configs/snli_project1_frozen.json](configs/snli_project1_frozen.json) records the
selected training recipe and group budget for reported CV, used in section 6.
Pass it explicitly with `--config` when preparing comparison folds/features.

Setup and the corresponding pipeline stages create these local, Git-ignored paths:

| Path | Purpose |
| --- | --- |
| `part1/data/huggingface/` | Cached source dataset |
| `part1/data/snli/<revision>/cleaned-v1/` | Cleaned official splits with original row identifiers; load later with `datasets.load_from_disk` |
| `part1/results/project1/manifests/setup.json` | Latest setup configuration, versions, interpreter, device, and Git state |
| `part1/results/project1/manifests/data_inspection.json` | Source/cleaned fingerprints, cleaning policy, counts, and artifact paths |
| `part1/results/project1/manifests/environment.txt` | Package snapshot from the current execution environment |
| `part1/results/project1/figures/` | Class-balance and training-length figures; later experiment figures |
| `part1/results/project1/metrics/data_audit/` | Split/class/length summaries, per-row lengths, exclusion IDs/reasons, duplicate groups, overlap counts, and training examples |
| `part1/results/project1/metrics/` | Later experiment scores alongside the data audit |
| `part1/results/project1/predictions/` | Later held-out predictions |
| `part1/results/project1/checkpoints/` | Later model artifacts |
| `part1/results/project1/checkpoints/implementation_smoke/run_*/` | Synthetic-only model checks and complete prediction bundles, separate from future SNLI models |
| `part1/results/project1/pilot/<pilot_id>/` | Pilot-only features/rows, candidate and selected model bundles, predictions, curves, metrics, timings, memory and budget projections |
| `part1/results/project1/manifests/pilot.json` | Pointer to the completed checked development pilot |
| `part1/results/project1/folds/<protocol_id>/` | Frozen `protocol.json`, `fold_assignments.parquet`, `pool_membership.parquet`, `excluded_rows.csv`, `pool_summary.csv`, and `fold_summary.csv` |
| `part1/results/project1/manifests/evaluation_design.json` | Pointer to the current frozen evaluation protocol |
| `part1/results/project1/features/<protocol_id>/seed_<seed>_fold_<fold>/<feature_id>/` | Tokenizer/inventory, ragged token arrays, row order/labels, canonical extended CSR counts, and checked feature manifest |

Setup only reads the pinned dataset revision from the configuration; it does
not contact Hugging Face or download SNLI. We resolved the revision through
Hugging Face's metadata API when creating the configuration.

`python -m part1.snli_data` performs the download and audit. It only excludes
invalid labels and unusable text; duplicate pairs and official-split overlaps
are retained in the saved reference splits. Its held-out-split inspection is
structural: there are no test predictions or test-driven model comparisons.
Whitespace-token lengths are exploratory counts, not learned BPE segmentation.

Rerunning setup refreshes its manifest. Per-experiment run histories will be
added with the training runner. A manifest with `repository.dirty = true`
records work in progress; commit the reviewed source and rerun setup when
recording a reproducible committed version.

Keep human explanations and results in [PROJECT1_FINDINGS.md](PROJECT1_FINDINGS.md)
and assistance disclosures in [AI_ASSISTANCE.md](AI_ASSISTANCE.md).
