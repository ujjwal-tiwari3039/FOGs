# FOGs: Bias-Resistant Short-Answer Assessment

## Challenge
Three-class short-answer grading (correct / contradictory / incorrect) from the SemEval 2013 Task 7 dataset, with a focus on **robustness** and **explainability** rather than raw accuracy alone.

## Deliberate Constraints
- **3-hour build time** (vs. the challenge's suggested 5–7 hours). This is a deliberate scope decision, not an oversight. See the scoring breakdown below for how it affected design choices.
- **No external LLM API calls at inference time.** The grading environment's internet access is unconfirmed. All model weights ship with the submission.
- **Internet access**: Available at build time. Models (sentence-transformers `all-MiniLM-L6-v2`) are downloaded and cached locally. Inference works fully offline once cached.

## Architecture

### Why not a fine-tuned LLM?
A LoRA-tuned 7B model would be a legitimate choice with 5–7 hours, but:
1. Fine-tuning alone consumes most of a 3-hour budget before touching robustness (55% of score)
2. Calibrating LLM probabilities for Brier score is an open research problem
3. Vendoring 7B+ weights for offline eval is fragile under time pressure

### What we built instead
1. **Canonicalization + boilerplate stripping** — makes CF1–CF3 inputs identical to originals by construction
2. **Sentence-embedding cosine similarity** (`all-MiniLM-L6-v2`) — primary signal, resistant to keyword stuffing (CF4)
3. **Multi-reference max-similarity** — handles the many questions with 2–14 reference answers
4. **Calibrated Logistic Regression** (`class_weight="balanced"`, Platt scaling) — honest probabilities for Brier score, handles 17% minority class
5. **Reference-embedding caching** — each question's references embedded once, not per student answer

### Features
- Max cosine similarity (student vs. best reference)
- Mean cosine similarity (student vs. all references)
- Question similarity (student vs. question text)
- Length features (student word count, reference word count, ratio)
- [Optional] NLI cross-encoder scores (contradiction/entailment/neutral)

## Scoring Target

| Component | Weight | How we address it |
|-----------|--------|-------------------|
| Macro-F1 | 45 pts | Balanced LR + sentence embeddings |
| CF1–CF3 consistency | 30 pts | Canonicalization is idempotent; boilerplate stripping |
| CF4 resistance | 15 pts | Embedding similarity, not word counts |
| Calibration (1−Brier) | 5 pts | Platt scaling via CalibratedClassifierCV |
| Reproducibility | 5 pts | Pinned seeds, requirements.txt, single-command run |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (train + evaluate + counterfactuals + scorecard + explanations)
python run_all.py

# Run with NLI cross-encoder (stretch feature)
python run_all.py --use-nli

# Run test-set generalization check (reads test/3way, outputs numbers only)
python local_eval.py
```

## Repository Layout

```
ed05-submission/
  data/                    # Cloned SemEval repo (not committed)
  src/
    load_data.py           # XML loader for training/3way/** only
    preprocess.py          # Canonicalization + boilerplate stripping
    counterfactuals.py     # CF1–CF4 generators
    features.py            # Sentence embedding cosine similarity + caching
    model.py               # Calibrated Logistic Regression
    metrics.py             # All rubric metrics + guardrail checks
    explain.py             # Evidence strings + confidence flags
  models/                  # Saved model artifacts
  local_eval.py            # ONLY script that touches test/3way/**
  run_all.py               # Single-command full pipeline
  requirements.txt         # Pinned dependency versions
  README.md                # This file
  BIAS_ANALYSIS.md         # Limitations and bias analysis
  EXPLANATIONS.md          # Generated example predictions
```

## Data Integrity

- `training/3way/**` feeds the model and CF4 vectorizer
- `test/3way/**` is ONLY read by `local_eval.py`, whose only output is printed numbers
- No gold labels are loaded, queried, or memorized in any training/feature code path
- Reference answers are used unconditionally regardless of `category` attribute (which is absent for all sciEntsBank data)

## Random Seeds
All random seeds are fixed to 42:
- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)` (if available)
- `sklearn` random_state parameters

## Expected Runtime
- Full pipeline (`run_all.py`): ~2–5 minutes on a machine with GPU
- Test evaluation (`local_eval.py`): ~1–2 minutes

## Assumptions
1. The grading environment may not have internet — models must work offline once cached
2. `all-MiniLM-L6-v2` (22M params) is small enough to vendor reliably
3. The evaluator generates its own CF1–CF4 variants and calls our inference code directly
4. Production scaling: reference-embedding caching means marginal per-submission cost is one embedding pass, not two
