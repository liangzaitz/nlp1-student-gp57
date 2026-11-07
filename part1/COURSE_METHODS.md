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
2. Use one shared 2,000-token BPE trained on eligible training + validation
   text, as requested by the group. CV scores are conditional on that tokenizer,
   which sees CV held-out text without labels; no test text enters its fitting.
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
| Model versus decision rule | CLASS 8-9, 55; T1 Text Classification | Compute a three-class pmf and the separate argmax decision in the visible code |
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
| Random seeds, CV, multiple runs, variability | CLASS 47; T1 Experiment; P1 | Use paired fold assignments and fixed seeds; plot all five fold scores |
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
order sensitivity. With `V=B=2000`, all three inputs have up to 4,000 coordinates
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

## Current implementation and review

The active experiment is entirely in [practical_1.ipynb](practical_1.ipynb).
A single fixed-setting flow replaces the historical pilot, search and artifact
machinery. The lecture references above remain the basis for the methods.

| Component | Current notebook evidence |
| --- | --- |
| Data and evaluation | Section 2: 559,164 eligible train+validation pairs; 60,002-row whole-group CV subset; five grouped folds, seed 0 |
| BPE and features | Section 3: one shared 2k BPE; fold-training bigram inventories capped at 2k; role-separated sparse inputs |
| Logistic and neural classifiers | Section 4: softmax/log-softmax, sparse first layer, ReLU, dropout, explicit minibatches/autodiff/AdamW |
| Fixed schedule | Three neural epochs in CV and final refit; no checkpoint selection or hyperparameter search |
| Metrics and selection | Section 5: all nine candidates; highest mean accuracy, lower NLL for exact ties; macro-F1 and descriptive fold SD |
| Bayesian analysis | Section 6: five paired scores, correlated-t model, 18 one-factor comparisons and ROPE sensitivity |
| Error review | Section 7: one OOF repetition; section 8: test confusions and representative errors; students must interpret these |
| Final evaluation | Section 8: refit selected classifier on full development pool, fixed shared BPE, official-test metrics and graphs |
| Tutorial-style plots | Section 9: boxplots with fold points, posterior mean-difference curves and ROPE sensitivity heatmap; PNG/PDF exports |
| Literal code | All implementation in notebook cells; no local helper imports, external configuration/result reads or dynamic code loading |

Premise grouping, sentence-role concatenation, the fixed CV subset, feature caps
and the globally fitted tokenizer are task-specific design choices. The shared
BPE means this CV evaluates classifiers conditional on development-fitted
preprocessing. Five dependent folds provide limited posterior information.
Earlier experiments already used test; the current report discloses that history.
The current result must not be presented as a controlled vocabulary-size ablation.

The notebook tests check leakage boundaries, feature counts/roles, complete groups,
fixed epochs, unchanged BPE during CV, automatic selection and posterior direction.
Students still need to verify error interpretations and explain the code.

## Course organization

INTRO pages 7–11 describe the three-project sequence, the 15-minute oral,
the same task/dataset across projects and the later 1,500-word blog. P1 allows
up to six optional backup slides. Group review and oral preparation remain;
check deadlines and slots against the official course schedule separately.
