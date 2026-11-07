# Historical configurations

The active experiment keeps its fixed settings in the `BASE` cell of
[practical_1.ipynb](../practical_1.ipynb): shared **2,000-token BPE**,
**2,000 bigram types**, one five-fold CV pass and three neural epochs.
It does not read the JSON files in this directory.

These files remain references for earlier experiments and the retained Python modules:

| Files | Historical use |
| --- | --- |
| `snli_project1.json`, `snli_pilot.json` | Initial setup and engineering pilot |
| `snli_project1_frozen.json` | Earlier combined unigram/bigram representation |
| `snli_project1_frozen_bigram.json` | Fold-trained 5k BPE, pure bigrams, three CV repetitions |
| `snli_tuning.json` | Earlier validation search, removed from the active workflow |
| `snli_final.json` | Earlier 5k unigram MLP1, five epochs and its separate test result |

Do not attribute these historical configurations or their test scores to the
current shared-2k experiment. See the notebook and [current findings](../PROJECT1_FINDINGS.md).
