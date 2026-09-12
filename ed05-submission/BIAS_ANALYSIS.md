# ED-05 Bias Analysis & Limitations

## Performance by Domain

> Results below are computed from `local_eval.py` against the held-out `test/3way/**` data.
> This script is the ONLY code path that reads test data, and its only output is printed numbers.

### Beetle vs. SciEntsBank

| Domain | Macro-F1 | Notes |
|--------|----------|-------|
| Beetle | 0.3971 | Electronics tutoring dialogues, single domain |
| SciEntsBank | 0.4223 | Short-answer science questions, multiple domains |

**Expected gap**: Beetle questions tend to be more formulaic (circuit descriptions), while SciEntsBank spans many science domains. A performance gap here is a real bias finding — the model may rely on domain-specific vocabulary patterns rather than genuine semantic understanding.

### Per-Condition Generalization

| Condition | Macro-F1 | N |
|-----------|----------|---|
| unseen-answers | 0.4197 | 979 |
| unseen-questions | 0.4249 | 1552 |
| unseen-domains (SciEntsBank only) | 0.4211 | 4562 |

**Interpretation**: unseen-domains is the hardest condition — the model must generalize to entirely new scientific topics. A significant F1 drop from unseen-answers → unseen-domains would indicate the model is partially memorizing domain vocabulary rather than learning answer-grading logic.

## Per-Class Analysis

| Class | F1 | Recall | Precision | % of data |
|-------|-----|--------|-----------|-----------|
| correct | 0.62 | 0.67 | 0.59 | ~42% |
| incorrect | 0.63 | 0.68 | 0.60 | ~45% |
| contradictory | 0.01 | 0.01 | 0.18 | ~13% |

### Why `contradictory` is structurally the hardest class

The 3-way label scheme collapses the original 5-way labels: `correct` and `contradictory` are unchanged, while `incorrect` = `partially_correct_incomplete` + `irrelevant` + `non_domain`.

**`contradictory` is specifically about negation/opposition**, and cosine similarity alone is bad at catching it. Two answers can share almost every content word and still be a contradiction:
- Reference: "the bulbs will stay lit"
- Student: "the bulbs will go out"

Embedding similarity scores this pair as fairly close, since it's dominated by topical overlap, not polarity. This is the actual mechanism behind `contradictory` being both the minority class (~17%) and typically the lowest-F1 class.

The NLI cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`), if enabled, directly addresses this by providing explicit contradiction/entailment/neutral scores trained on natural language inference data.

### The `incorrect` bucket is heterogeneous

`incorrect` merges:
- **Partially-correct-incomplete**: high overlap with reference, just missing key details
- **Irrelevant**: off-topic responses
- **Non-domain**: responses outside the subject area entirely

This means the decision boundary for `incorrect` is messier than for the other classes — some "incorrect" answers are semantically close to the reference (partial credit) while others are maximally distant (irrelevant).

## Counterfactual Robustness

| Metric | Score | Notes |
|--------|-------|-------|
| CF1–CF3 Consistency | 1.000 | Should be near 1.0 due to canonicalization |
| CF4 Resistance (U_CF4) | 0.9815 | Embedding similarity resists keyword stuffing |

**Design rationale**: CF1 consistency is achieved by construction — canonicalization is idempotent, so an already-canonicalized original and its CF1 variant become identical inputs. CF2/CF3 boilerplate strings are stripped before feature extraction. CF4 resistance comes from using embedding cosine similarity rather than raw keyword overlap as the primary signal.

## Calibration

Normalized Brier score: 0.2676 (Calibration Score: 0.7324)

Probability calibration uses Platt scaling (sigmoid) via `CalibratedClassifierCV` with 5-fold cross-validation. The `1/(2N)` normalizer keeps the score in [0, 1].

## Known Limitations

1. **Beyond CF1–CF4**: The four specified counterfactual families are narrow. The model has NOT been tested against:
   - Genuine paraphrasing (semantic-preserving rewrites beyond simple style changes)
   - Second-language phrasing patterns (non-native English constructions)
   - Adversarial inputs designed to exploit embedding space weaknesses

2. **Negation detection**: Without the NLI cross-encoder, negation detection falls back to keyword matching (e.g., looking for "not", "no", "never"). This misses:
   - Double negatives ("not disconnected" = connected)
   - Implicit negation ("failed to connect" vs. explicit "not connected")
   - Context-dependent negation

3. **Domain dependence**: The sentence embedder (`all-MiniLM-L6-v2`) is a general-purpose model, not fine-tuned for educational assessment. Performance on highly specialized scientific terminology may be weaker.

4. **Production scaling**: Reference-embedding caching means marginal per-submission cost is one embedding pass (the student answer), not two. This is the honest answer for scaling beyond a hackathon-sized test set. The embedding model itself is 22M parameters and runs in milliseconds per sentence.

## Data Integrity Statement

- `training/3way/**` feeds the model and the CF4 TF-IDF vectorizer
- `test/3way/**` is read ONLY by `local_eval.py`
- `local_eval.py`'s only output is printed numbers — nothing feeds back into training, features, or thresholds
- No evaluation gold labels or the public corpus's answer key were loaded, queried, or memorized anywhere in the training or feature code path
