# Project 1 configurations

- `snli_project1.json`: current reference/pilot input. Representations are
  `indicators`, `unigram_counts`, and `bigram_counts`; neural epochs are unset
  until the measured pilot selects them.
- `snli_pilot.json`: fixed group sampling, candidate epochs, solver limits and
  runtime planning policy.
- **`snli_project1_frozen_bigram.json`**: current selected recipe for reported
  cross-validation. Use it explicitly with `--config` for comparison folds and
  features. It records the refreshed bigram-only pilot and the fixed group budget.
- `snli_project1_frozen.json`: historical combined unigram/bigram recipe from
  the first pilot. Retained unchanged so its archived report still validates;
  it is not the current recipe. Historical `v1` bundles require the earlier code.

Current bigram input is `[premise bigram counts; hypothesis bigram counts]`,
dimension `2B`. With `V=B=5000`, all three current representations have 10,000
coordinates. See the executed notebook and `PROJECT1_FINDINGS.md` for measurements.

Step 7's completed run is `713ab1f5b3cb5801` (135 primary plus 30 baseline fits).
The frozen configuration remains unchanged: its status records when settings
were frozen, while the separate CV manifest records execution completion.

Step 8 uses `snli_analysis.json` for the comparison set, interval mass and
optional ROPE sensitivity. Its primary ROPE is read from the unchanged frozen
training configuration, not selected from results. Analysis source/CV/policy
hashes identify separate report directories; training is not repeated.

`snli_tuning.json` is now the **900-second default**: three explicit joint
recipes for each system, one seed, a fixed 100k-row complete-group training sample,
all eligible validation rows, patience 3 and at most 8 epochs. It may reference
compatible observations from the stopped large search, with explicit lineage
and compatibility checks. It has no full-data/extra-seed/refinement stage.

`snli_tuning_extended.json` preserves the previous extensive policy: 216 coarse
trials, refinement and full-training/three-seed finalists. Its run was stopped
when the group requested roughly 15 minutes. It is optional, not the default.
Neither tuning policy overwrites frozen baseline configurations or scores test.

The streamlined main notebook reads the frozen baseline and short-search policy
in setup. Its `run_search` is a simple fresh/resumed search in a separate
`results/project1/streamlined/` directory. It uses candidate overrides, epoch cap,
patience and the time limit; legacy import, worker and extended-stage settings
belong to [the detailed notebook](../practical_1_details.ipynb). It never rewrites
the completed historical run or silently imports its trials. `RUN_SEARCH` and
`RUN_CV` default to false; default Run All checks existing artifacts.

`snli_final.json` freezes the Step 9 final choice before any official test score:
5k BPE/unigram counts/128-unit MLP1, seed 0, five epochs selected by validation
accuracy for candidate `29aa244710c5fabe`. It records the 559,164-row final pool,
900-second allowance, source experiment IDs and selection limitations. The final
fit/test are complete in `results/project1/final/snli_5k_unigram_mlp1/`; the existing
three-epoch baseline configuration is unchanged. The selection record remains
pre-test history; the final manifest supplies current completion/metrics.
Do not score a partial final fit if the complete frozen schedule cannot finish.
