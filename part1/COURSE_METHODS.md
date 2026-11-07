# Course methods and their role in Project 1

Reviewed on 2026-11-05. This file tracks the current material supplied by the
group and connects it to the SNLI implementation. Update the evidence/status
entries as work is completed; reading a method is not implementing it.

The explicit Project 1 requirements come from `project.ipynb`. Lecture topics
help us choose and explain methods, but their presence in a deck does not make
every topic an additional implementation requirement. Practical 1's tutorial is
also a primary course source (Introduction, PDF page 11).

## Source index and page numbering

Page numbers below are **physical PDF pages, starting at 1**, including overlay
pages. The decks' printed slide numbers differ: for example, HC2a PDF page 34
shows printed slide 15. All four decks were read, including overlay/reference
pages; selected structural/model diagrams were also inspected as images.

| ID | Supplied material | PDF pages | Main contribution |
| --- | --- | --- | --- |
| INTRO | [HC1a-Introduction.pdf](/home/borisv/Downloads/HC1a-Introduction.pdf) | 35 | Course structure, projects, oral assessment, research context |
| LANG | [HC1a-How-Language-Works.pdf](/home/borisv/Downloads/HC1a-How-Language-Works.pdf) | 65 | Lexicon, productivity, morphology, syntax, linguistic dependencies |
| CLASS | [HC2a-Text-Classification.pdf](/home/borisv/Downloads/HC2a-Text-Classification.pdf) | 58 | Fixed features, softmax classifiers, estimation, evaluation, limitations |
| GEN | [HC2b-Text-Generation.pdf](/home/borisv/Downloads/HC2b-Text-Generation.pdf) | 73 | Language-model factorization, n-grams, neural histories, generation evaluation |
| T1 | [tutorial.ipynb](tutorial.ipynb) and supplied Python modules | Notebook sections | Digital text, BPE, classifiers, CV, Bayesian comparison, ensembles |
| P1 | [project.ipynb](project.ipynb) | Checklist and Get to work sections | Assignment requirements and oral preparation |

The PDFs remain at the supplied local paths; they have not been copied into the
repository. For sharing, refer to the authorized course copies. Their text is
study material, not a request to carry out their administrative action items.

## Decisions for this implementation

1. Keep SNLI as three-way classification with input `x = (P, H)`.
2. Keep the T1 BPE tokenizer as the common segmentation procedure.
3. Compare binary unigram indicators, raw unigram counts, and bigram-only
   counts. The third choice follows CLASS pages 24 and 49-50 and T1's
   bag-of-bigrams discussion. On 2026-11-06 the group requested pure bigram
   counts in place of the earlier combined unigram/bigram representation.
4. Compare log-linear softmax regression, the T1 one-hidden-layer ReLU MLP, and a
   two-hidden-layer ReLU MLP. Changing depth is our small architecture extension,
   supported by CLASS pages 31-35 and allowed by P1.
5. Use argmax for ordinary decisions. Train by negative log-likelihood and
   assess both probabilities (held-out NLL) and decisions (accuracy/macro-F1).
6. Retain paired cross-validation and Bayesian estimation with a motivated ROPE,
   as explicitly required by P1 and demonstrated in T1.
7. Keep lecture methods beyond this design as explanations or optional
   extensions, rather than silently expanding the assignment.

## Classification methods to implement and explain

Everything in this table is planned for the SNLI pipeline. The Evidence column
identifies what must demonstrate completion; it does not assert existing results.

| Method/concept | Source | Planned use and evidence |
| --- | --- | --- |
| Model versus decision rule | CLASS 8-9, 55; T1 Text Classification | Save a three-class pmf and the separate argmax decision; show one worked example |
| Fixed feature map `h(x)` in `R^D` | CLASS 18-24; T1 Text Representation | Define feature coordinates, role blocks, dimension, and information loss |
| Binary token indicators | CLASS 20-21; T1 TokenIndicator; P1 | First representation; manually check presence values |
| Unigram counts / bag-of-words | CLASS 22-23; T1 Bag-of-Words exercise; P1 | Second representation; manually check repeated-token counts |
| Ordered multi-token / n-gram features | CLASS 24, 49-50; T1 Named Axes questions | Third representation: counts of retained adjacent BPE-token bigrams only |
| Text normalization and information loss | CLASS 24 | Document casing/punctuation choices and preserve negation; inspect examples |
| Log-linear logits and softmax | CLASS 26-29; T1 loglin.py; P1 | Fit multiclass logistic regression; explain weights, biases, normalized probabilities |
| ReLU feed-forward network | CLASS 31-34; T1 smlp.py; P1 | Fit the one-hidden-layer MLP with fixed sparse inputs |
| Width/depth as design choices | CLASS 35; P1 | Add the two-hidden-layer MLP; record parameter counts and compare on identical folds |
| Conditional maximum likelihood / NLL | CLASS 37-43 | Explain `-log p_theta(y\|x)` and mean batch loss; verify held-out NLL |
| Mini-batches, autodiff, gradient updates | CLASS 41-43; T1 smlp.py | Explain forward pass, backward pass, and optimizer step in the training loop |
| Weight decay and dropout | CLASS 44; T1 smlp.py | Record regularization settings; enable dropout for training and disable it for evaluation |
| Probability-level versus decision-level evaluation | CLASS 46, 51-55 | Report NLL as well as accuracy/macro-F1; show why one can change without the other |
| Random seeds, CV, multiple runs, variability | CLASS 47; T1 Experiment; P1 | Save paired fold assignments and seeds; plot variation and explain its sources |
| Bayesian comparison and ROPE | T1 Data analysis model; P1 | Compare paired fold accuracies, report posterior uncertainty and practical-equivalence probabilities |
| Representation/argmax/label limitations | CLASS 49-54; LANG 20-35 | Analyze errors and ambiguous examples; explain collisions and uncertainty |

BPE and Bayesian analysis are developed in T1, rather than fully taught in the
four slide decks. They remain in scope because P1/T1 explicitly cover them.
AdamW is the optimizer used by the tutorial; the slide decks do not prescribe its
particular hyperparameters. Report AdamW's actual decoupled weight decay rather
than claiming it is identical to every implementation of an L2 penalty.

## Working mathematical specification

The input is a premise/hypothesis pair `x = (P, H)`, and `K = 3`.

- Encode the sentences separately and concatenate two fixed-size role blocks:
  `h(x) = [v(P); v(H)]`. This is our SNLI adaptation of the fixed feature map.
- Indicators and unigram counts have pair dimension `D = 2V`.
- With `B` retained bigram types, bigram-only counts have dimension
  `D = 2B` and layout `[P bigrams; H bigrams]`. Learn the inventory only from training data. Count
  adjacent BPE tokens within each sentence, without crossing the role boundary.
- BPE tokens are subword units. A bigram of these units need not be two complete
  linguistic words; keep this distinction explicit in examples and conclusions.
- Log-linear model: `s(x) = W h(x) + b`; parameter count `KD + K`.
- One-hidden-layer model: `s(x) = W2 ReLU(W1 h(x) + b1) + b2`, with training
  dropout. Parameter count `DH + H + HK + K`.
- Two-hidden-layer model: insert another affine/ReLU layer, with training
  dropout; record its width and resulting parameter count too.
- Both families prescribe `p_theta(k|x) = exp(s_k) / sum_j exp(s_j)`.
- Decisions use `argmax_k p_theta(k|x)` with a documented tie convention.
- The data loss is `L(theta) = -(1/n) sum_i log p_theta(y_i|x_i)`, with the
  chosen regularization added or applied through the optimizer as appropriate.
- Use the tutorial's `LogSoftmax` plus `NLLLoss`, or raw logits plus
  `CrossEntropyLoss`. Apply the matching convention consistently.
- Accuracy measures the fraction of correct decisions. NLL evaluates the
  probabilities assigned to gold labels. Macro-F1 gives each class equal weight.

Replacing unigrams with bigrams changes feature identities, sparsity and local
order sensitivity. With `V=B=5000`, all three inputs have 10,000 coordinates
and matched parameter counts at each architecture. Equal width does not make
their information equivalent: one-token sentences and sentences with no retained
adjacencies produce zero bigram blocks. Individual token counts are omitted;
there is no unigram fallback. Bigrams still do not encode a full parse tree.

## Linguistic concepts to use in interpretation

| Concept | Source | Connection to SNLI / oral explanation |
| --- | --- | --- |
| Conventional lexical meaning and listemes | LANG 9-13, 41-46 | A token identity is not a complete semantic representation; memorization alone is insufficient |
| Productivity and combinations | LANG 12-18 | Explain generalization to previously unseen sentence combinations |
| Meaning depends on order and roles | LANG 1-5, 20, 28-35 | Motivate premise/hypothesis role blocks and order-sensitive error categories |
| Hierarchy and long-range dependencies | LANG 23-37 | Explain what unigrams and local bigrams omit; discuss parsing as context, not a required model |
| Morphology, morphemes, roots/stems | LANG 39-44, 59; T1 Digital Text | Explain why segmentation matters, and why BPE units need not equal morphemes |
| Learning biases and limited observations | LANG 47-60 | Discuss why data-driven associations do not establish complete language understanding |
| Data provenance, resources, and broader impact | INTRO 27-29; GEN 69 | State dataset/annotation/domain limits and avoid overstating conclusions |

For order diagnostics, inspect the actual tokenized vectors rather than assuming
the encodings collide. A role-swapped sentence is not automatically a
contradiction: manually reason about the NLI label and preserve dataset labels.

## Discussed feature and decision alternatives

These are useful explanations or later experiments, not extra required systems.

| Alternative | Source | Current treatment |
| --- | --- | --- |
| Stemming/root-based features | CLASS 24; LANG 39-44 | Explain the motivation and possible loss of distinctions; do not add to the core grid |
| Collocation detection | CLASS 24 | Distinguish statistically associated expressions from simply adjacent bigrams |
| Negation-scope marking | CLASS 24, 50 | Mention as a possible next feature change; ordinary bigrams do not solve general negation scope |
| Additional normalization/placeholders | CLASS 24 | Document current policy; consider only as a controlled future change |
| Expected-utility decisions | CLASS 53, explicitly optional | Explain as an alternative to argmax; no implementation required |
| Selective prediction, confidence/correctness analysis, AUROC | CLASS 53, explicitly optional | Keep as optional evaluation extensions |
| Test-time dropout / predictive uncertainty | CLASS 44, advanced optional | Do not confuse with the mandatory Bayesian comparison of experimental scores |
| Human label variation | CLASS 54 | Discuss ambiguous annotations without treating maximum softmax probability as human agreement |
| TF-IDF | Earlier proposed plan; not explicitly developed in these decks | Optional future representation; replaced by a directly covered n-gram feature design |
| Whole-document tabular CPDs | CLASS 11-17 | Explain lack of parameter sharing and generalization; no need to build an extra system |
| Naive Bayes | CLASS 12-14, footnote | Mentioned as a different tabular approach; not one of P1's required architectures |
| Ensembles / majority voting | T1 System Combination | Optional after core comparisons; not the extra architecture required by P1 |

## Text-generation methods: covered, but separate from the SNLI core

| Method/concept | GEN pages | What we should understand |
| --- | --- | --- |
| LM as a distribution over finite token strings | 6-14 | Sequence probability/sample space versus a three-class conditional label distribution |
| Autoregressive chain-rule factorization | 16-29, 31 | Product of next-token probabilities; history, EOS, and sequential sampling |
| Relative-frequency categorical estimates | 32-36 | How observed counts estimate probabilities and why unseen events/history cause sparsity |
| Unigram/bigram/trigram LMs and Markov assumptions | 37-42 | Truncated histories, conditional independence, parameter growth, long-range limitations |
| Neural next-token softmax | 43-50 | The same logits/softmax idea, now over `V` vocabulary outcomes rather than `K` NLI labels |
| Indicators and trainable word vectors | 44-45 | Token representations; trainable vectors are introduced, not a requirement to use pretrained embeddings |
| Fixed-history composition: averaging, concatenation, FFNN | 46-47 | Input/output dimensionality and order trade-offs |
| Variable-history concatenation versus pooling | 49-54 | Concatenation grows with history length; mean pooling stays fixed-size but loses order |
| Basic recurrent composition | 55 | Sequential state update `h_t = tanh(U e_t + R h_(t-1) + b)` and reused parameters |
| RNN limitations; GRU/LSTM direction | 56, 63-64 | Vanishing/exploding gradients; detailed cells and practical recurrent dropout deferred to HC3b/T2 |
| LM likelihood and gradient training | 58-64 | Sum contributions across positions and average across text examples; share training principles with classification |
| LM evaluation: NLL/perplexity, extrinsic tasks, generated statistics | 66-68 | Choose metrics for the modeled outcome space; do not report language-model perplexity as SNLI classification performance |
| Model/data assumptions | 69 | Interpret scores and generated behavior in light of training objective and dataset limitations |

We will explain connections from HC2b when helpful, while keeping the required P1
grid on fixed sparse features and feed-forward/log-linear classifiers. Sequence
encoders can be revisited for Project 2; LLM methods belong to the later tutorial
sequence described in INTRO page 8. A linguistic generative grammar (LANG 21) is
also a different concept from a probabilistic text generator.

## Experimental choices we added for SNLI

The following support fair, feasible experiments, but are not extra lecture
methods that every student must implement:

- Premise-grouped folds, because examples can share premises.
- Concatenated premise/hypothesis feature blocks, because the input is a pair.
- CSR storage and sparse first-layer multiplication, following the tutorial's
  sparse direction and avoiding large dense intermediate matrices.
- All nine feature/architecture combinations, five folds, and three repetitions.
- A majority/class-prior baseline and a hypothesis-only diagnostic.
- A pilot-based budget, a documented fixed subset if needed, checkpoints,
  stable example IDs, saved predictions, and a final untouched test evaluation.

We will justify these choices in the notebook. For instance, five-fold CV and
Bayesian comparison are course methods, while grouping those folds by SNLI
premise is our task-specific adaptation.

## Implementation and understanding tracker

All four supplied lecture decks have been reviewed. Baseline Steps 1–8 and the
short validation search, Step 9 error review/final design and Step 10 final fit/test
are complete. New tuned CV remains optional; group review/oral preparation are next.
The main notebook was simplified on 2026-11-06 while retaining these methods.
The historical pilot, extensive search and detailed resume machinery are in
[practical_1_details.ipynb](practical_1_details.ipynb); the main notebook never loads it.

| Component | Current status and evidence |
| --- | --- |
| Environment, seeds, data provenance | Notebook sections 1–3; pinned SNLI revision, saved source IDs, [environment](environment.txt), [setup](README.md) |
| Task, labels and data examples | Section 2; counts, labelled pairs, training-only label/length plots; [data tests](tests/test_snli_data.py) |
| BPE and train-only preprocessing | Section 4; separate vocabulary/inventory per fold, actual BPE examples, saved mappings |
| Indicators, unigram and pure bigram counts | Section 4; explicit sparse role blocks, repeated-token values, controlled order permutation; [feature tests](tests/test_practical_1_notebook.py) |
| Model → pmf → decision | Section 5; visible logits/softmax/argmax calculation on an illustrative unlabelled pair |
| Log-linear and one/two-hidden-layer MLPs | Section 5; literal classes/training function, sparse first multiplication, ReLU/dropout and raw-text bundle reload |
| MLE/NLL, gradients, regularization | Section 5; explicit minibatch/autodiff/AdamW loop and 30-update teaching curve; section 6 shows validation NLL overfitting |
| NLL versus accuracy/F1 | Sections 5–7; independent probability-based metric recomputation; accuracy-selected checkpoints need not minimize NLL |
| Historical fixed-epoch baseline | Frozen [configuration](configs/snli_project1_frozen_bigram.json); three epochs before reported CV; outer scores never select checkpoints |
| Repeated grouped CV | Sections 3/7; 60,005 rows, five folds × three split seeds, 165 fits; complete-group boundaries and paired row IDs checked |
| Bayesian comparison and ROPE | Section 8; direct corrected-t computation, 28 contrasts, 95% intervals, 84 sensitivity rows, preselected ±0.005 accuracy ROPE |
| Joint tokenizer/model tuning | Section 6; 27 completed observations, nine reused/18 new, 7m 58s; 100,001 train/9,839 validation, one seed, three joint recipes; [policy](configs/snli_tuning.json) |
| Code in notebook cells | 42 code cells / 1,297 code lines; fresh-kernel Run All with project imports blocked; no dynamic helper-file execution; [literal-definition tests](tests/test_practical_1_notebook.py) |
| Practical-style presentation | Short explanations, worked token/feature/model inputs, explicit gradient updates, comparative plots and qualified interpretations |
| Error analysis and final design | Completed first pass: section 9, 60,005 unique OOF rows, class confusions, overlap/negation slices, shared errors and 50 distinct annotated cases; 11 text/label flags retained, student review still required; [frozen final recipe](configs/snli_final.json), five epochs chosen on validation |
| Final test | Completed Step 10: 559,164 refit rows, five epochs, 2.98 minutes; 9,824 test pairs, accuracy 74.58%; one frozen-model prediction pass, saved probabilities/bundle, four graph sets |
| Oral preparation | Pending Step 11; group review and optional backup slides still required |
| HC2b concepts | Reviewed; connections only, no sequence model added to P1 |

Verification: 69 repository tests passed; main Run All independently reproduced
all saved CV/validation metrics and Bayesian/ROPE tables. The new fixed-training
loop matches the earlier implementation on synthetic cases. These are software
checks, not evidence that students have reviewed or can explain every method.
See [current findings](PROJECT1_FINDINGS.md) for exact scores, scope and limitations.

## Course organization to keep in mind

INTRO pages 7-11 explain the three-project sequence, a 15-minute oral assessment,
the same task/dataset across projects, and the later 1,500-word blog. P1's
specific checklist allows up to six backup slides rather than a presentation;
the overview mentions a handful of slides and the project-repository URL. We
will prepare the compact backup deck and URL while following the specific P1
requirements. Keep deadline/slot verification separate from this implementation
plan and use the official course schedule for current details.
