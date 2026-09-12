# MASTER BUILD PROMPT — ED-05: Bias-Resistant Short-Answer Assessment (v2)

Paste this whole document as your first message to the Antigravity agent. It's written as a complete spec so it can go straight into Planning Mode as a task list. If the hackathon has published any FAQ/clarification that contradicts an assumption below, follow the FAQ instead.

## Before the clock starts
- **Confirm Antigravity's Terminal Command Auto-Execution policy is set to auto-run** (Antigravity → Advanced Settings). If it's left on manual-approve, every `git clone`, `pip install`, and script run below stops and waits for a click — on a 3-hour budget that alone can eat a real chunk of your time.
- **Check your quota tier.** Antigravity's free tier refreshes on a rolling ~5-hour window, and Planning-Mode work (this entire build) draws it down faster than simple edits. If you're on a metered/shared quota, decide now what you'd cut first if the agent stalls mid-build (see "cut stretch items first" below) rather than discovering it at hour two.
- **This challenge's own header says "Expected Time: 5–7 hours"; you're building to 3.** That's a legitimate choice, but state it as a deliberate constraint in the README, not a silent one — it changes what "done" reasonably looks like, and a judge reading the README should know why scope was cut.

## Role
Act as a senior ML engineer pair-programming a hackathon submission under a hard **3-hour** deadline. Prioritize a small, fully-working, robust system over an ambitious one that doesn't finish. Make reasonable assumptions and document them in the README instead of stopping to ask questions.

## What we're building
A three-class short-answer grader for "ED-05 — Bias-Resistant Short-Answer Assessment" (Responsible & Explainable AI in EdTech). Given a question/context, a reference answer (**when supplied — see fallback below**), and a student's answer, output calibrated probabilities over {correct, contradictory, incorrect}. The system must judge *meaning*, not *phrasing*: it should give the same verdict under harmless style changes, and it must not be tricked into a higher "correct" score just because extra keywords are stapled onto an answer.

- **Reference answer may be missing.** The Final Evaluation section's own wording — "the reference answer where supplied by the task data" — is conditional. A zero-reference input must not crash the pipeline; fall back to a question-only plausibility check (or a lower-confidence default away from "correct") instead of throwing.
- **Reference answer is often plural.** In the actual corpus (verified below), about a quarter of questions ship 2 or more reference answers — up to 14 in one case — tagged `category="BEST"` or `"MINIMAL"`. Don't hardcode "the reference"; take the **max similarity across all supplied references** (a student matching any one valid phrasing should count as matching).

## How this is scored — build in this priority order
- 45 pts — macro-F1 on the original three-way labels
- 30 pts — consistency under CF1–CF3 (style-only paraphrases of the same answer)
- 15 pts — resistance to CF4 (keyword-stuffing)
- 5 pts — calibration (normalized multiclass Brier score)
- 5 pts — reproducibility

**50 of 100 points are robustness/calibration, not raw accuracy — but that's not license to under-build the classifier.** A model that predicts one fixed class for every input scores close to full marks on consistency (30/30, it never changes its answer) and CF4 resistance (15/15, if it never predicts "correct" then `P_correct` never moves and `U_CF4` clips to ~1). That's ~45 of 100 points from a model with no real understanding, so:
- Track macro-F1 *alongside* the robustness metrics at every checkpoint, not as an afterthought once robustness looks good.
- Build a guardrail into the scorecard script: flag, loudly, if the predicted-class distribution on the validation split collapses to ≤2 classes, or if any class's recall is 0. High consistency/CF4-resistance paired with a collapsed distribution means the "robustness" is coming from indifference, not comprehension.

Build the robustness-by-design preprocessing below early — it's still genuinely cheap — but treat the guardrail above as equally load-bearing, not optional polish.

Also important: the evaluator generates its own CF1–CF4 variants on its own held-out examples and calls your inference code directly — it does not run your local counterfactual-generator script against them. So the robustness has to live *inside* the classifier's own input pipeline, not only in a separate test script.

## Data — verified against the actual repo, not assumed
1. Clone `https://github.com/myrosia/semeval-2013-task7` — this is the official repo, maintained by Myroslava Dzikovska, one of the task's original organizers, so treat it as authoritative rather than something to sanity-check for legitimacy. Unzip **`semeval-3way.zip`, not `semeval-5way.zip`.** A clean, pre-made 3-way split already exists; don't hand-roll a 5-way→3-way collapse yourselves, that's a needless source of label bugs.
2. Real layout inside the zip:
   ```
   training/3way/{beetle, sciEntsBank}/*.xml            (182 files total: 47 + 135)
   test/3way/beetle/{test-unseen-answers, test-unseen-questions}/*.xml
   test/3way/sciEntsBank/{test-unseen-answers, test-unseen-questions, test-unseen-domains}/*.xml
   ```
   `test-unseen-domains` only exists for sciEntsBank (Beetle is a single tutoring domain, so "unseen domain" doesn't apply). There's no separate `dev/` folder — carve your stratified local validation split out of `training/` yourselves, as planned.
3. Format is nested XML per question, not a flat CSV: each `<question>` has one-or-more `<referenceAnswer category="BEST|MINIMAL">` and several `<studentAnswer accuracy="correct|contradictory|incorrect">` children. A simple `xml.etree.ElementTree` walk handles it fine — just don't budget zero minutes assuming a one-line `pd.read_csv()`.
4. **Measured class balance, training/3way (n=8,910 student answers): correct 41.2%, incorrect 41.4%, contradictory 17.4%.** Test/3way skews similarly (correct 41.9%, incorrect 45.5%, contradictory 12.6%). Contradictory is a real minority class relative to the other two — `class_weight="balanced"` isn't a nice-to-have, and it's exactly the class a degenerate model (see the guardrail above) is most likely to quietly stop predicting.
5. **The test folders are not gold-label-free — verify and firewall this explicitly.** Every one of the 252 test/3way XML files carries the true `accuracy=` label inline, same as training. That's a *good* thing for your own generalization check — it gives you three real held-out conditions (unseen answers, unseen questions, and for sciEntsBank, unseen domains) instead of just a same-distribution carve-out. But it means the hard constraint below is one careless glob away from being violated: if your loader or the CF4 vectorizer-fitting code ever walks `test/**` the same way it walks `training/**`, you've pulled labeled examples into a path that touches the model. Keep the two hard-separated (see Hard Constraints).
6. If you can recover which subset an item came from (Beetle = electronics-tutoring dialogues; SciEntsBank = short-answer science questions), keep that as a domain column — you'll want it for the bias write-up later.

## Do this first: robustness-by-design (≈45 of 100 points)
Build these into the classifier's *own* preprocessing, so they apply automatically to anything the evaluator sends later, not just to your own test script:

1. **Canonicalize every input** (reference and student answer, at train and inference time): lowercase + strip punctuation via `[^\w\s]` removal — not the `string.punctuation` translation table, which disagrees with that regex on characters like underscore. Use the *same function* everywhere the model canonicalizes text, since a mismatch between "how the model canonicalizes" and "how CF1 strips punctuation" would quietly undercut the idempotence this depends on. Hand-check it on 2–3 answers with contractions or hyphens (`"don't"`, `"single-pole"`) before trusting it. Done right, this is idempotent — an already-canonicalized original and its CF1 variant become identical inputs — which alone should push CF1 consistency close to 100%.
2. **Strip the two known boilerplate strings** (case-insensitively, with adjacent whitespace) before feature extraction: the prefix "in my answer, i think that" and the suffix "this is my final answer." These are published, fixed, deterministic strings, so stripping them is defensive preprocessing against a known nuisance transform — not label-fingerprinting. This collapses CF2/CF3 inputs back to the original text.
3. **Don't let raw keyword-overlap counts drive the "correct" score.** CF4 exists specifically to punish that shortcut. Make the primary signal a *sentence-embedding cosine similarity* between reference and student answer, **taking the max over all supplied reference answers** (see "Data" above) — mean-pooled embeddings dilute 1–2 extra trailing tokens far more than a raw overlap count would. If you also use lexical overlap features, cap them (presence/absence per key term, or a saturating transform) instead of raw counts.
4. **Cache reference-answer embeddings per question, computed once** — not recomputed per student answer. Every student answer to the same question shares the same handful of references, so this is the difference between one embedding per submission and one redundant extra embedding per submission. It costs nothing extra to build this way now, and it's the honest answer if a judge asks how this scales past a hackathon-sized test set.
5. (Stretch, only with time to spare) Add a cheap heuristic that down-weights a short 1–2 token fragment at the very end of an answer that doesn't grammatically connect to what came before — that's structurally what CF4 produces.
6. Once the classifier exists, run your own CF1–CF4 generator against your validation split *and* the guardrail check, and look at the actual numbers before calling this "done." Verify, don't assume steps 1–3 worked.

## Counterfactual generator — implement exactly, ambiguity noted inline
- **CF1**: `answer.lower()`, then strip punctuation via `[^\w\s]` removal — the same function used in the model's own canonicalization step above, so the two stay identical by construction.
- **CF2**: prepend exactly `"In my answer, I think that"`. The spec doesn't show whether a separating space is included before the original text — pick one convention, join with a single space, and use it consistently everywhere this string is referenced (including the stripping step above).
- **CF3**: append exactly `"This is my final answer."` — same note: pick one spacing convention and document it.
- **CF4**: Fit `sklearn.feature_extraction.text.TfidfVectorizer(lowercase=True, stop_words="english")` (default token pattern — don't override it) on the `training/3way` answers, stated explicitly in your README. For the current answer, transform it with this fitted vectorizer; "eligible tokens" are the vocabulary terms that get a nonzero weight for this answer. Take the top 2 by TF-IDF weight, breaking ties alphabetically ascending; append both once each, space-separated, at the end. If fewer than 2 eligible tokens exist, append all of them.
  - The spec doesn't state the relative order of the two appended tokens — pick one (e.g. descending weight) and note it.
  - The evaluator fits its own vectorizer on its own copy of the same public training answers, so the exact two words it appends won't always match what your local vectorizer would pick. That's fine — the mitigation above (embedding similarity, not word-count) doesn't depend on predicting *which* two words get appended, only on not being sensitive to *any* 1–2 extra trailing tokens. Don't over-fit CF4 resistance to your own vectorizer's specific picks.
- CF1–CF3 are pure style transforms used only for the consistency metric; CF4 is used only for the keyword-resistance metric. Don't optimize for CF4 "consistency" — optimize for `P(correct)` not moving up.

## Metrics — implement exactly as given, don't approximate
- Macro-F1 on the original three-way predictions: `sklearn.metrics.f1_score(y_true, y_pred, average="macro")`.
- Counterfactual consistency = fraction of all CF1+CF2+CF3 variants (pooled across types and examples) whose predicted class matches the model's prediction on the corresponding original answer.
- CF4 resistance utility: `U_CF4 = 1 - mean(max(0, P_correct(CF4_i) - P_correct(original_i)))`, clipped to [0, 1]. `P_correct` is the model's predicted probability specifically for the "correct" class — not the probability of whatever class it happened to predict.
- Normalized multiclass Brier score: `(1 / (2N)) * sum_i sum_k (p_ik - y_ik)^2` over the 3 classes. Note the `1/(2N)` normalizer (not the more common `1/N`) — that's what keeps it in [0, 1].
- **Guardrail (new):** print the predicted-class distribution and per-class recall on the validation split alongside the scorecard. If any class has 0 recall, or the distribution collapses to ≤2 classes, print a visible warning — this is the "cheap 45 points" trap from the scoring section, and it should be impossible to miss on the printout.
- Print a scorecard mirroring the real rubric: F1 → /45, consistency → /30, CF4 utility → /15, (1 − Brier) → /5, a reproducibility self-check → /5, and a total /100. Recompute after every material change so there's always a live estimated score.

## Explainability
For ~8–10 example predictions (spread across all three classes, plus at least one CF1–CF4 pair), output:
- predicted class + full probability vector,
- a short deterministic "evidence" string: key content words shared with the reference answer, key reference words missing from the student answer, and whether a negation/contradiction cue was detected — flag this last one as "possible negation," not a certainty, since naive negation-word matching misses double negatives and context.

If the final classifier is linear or tree-based on engineered features, also surface global feature importances (coefficients, or `shap.TreeExplainer` if there's a tree model and time) as a second, cheap layer of explanation.

## Limitations / bias analysis (short markdown file)
Compute these, don't just assert them:
- performance split by domain if recoverable (Beetle vs. SciEntsBank) — a gap here is a real bias finding, and squarely on-theme for an XAI-in-EdTech track,
- **a real held-out number, not just your carved split**: run the finished model, untouched, against `test/3way/**` and report macro-F1 separately for the unseen-answers, unseen-questions, and (sciEntsBank) unseen-domains conditions — a stronger generalization claim than a same-distribution carve-out, and the data supports it for free,
- how F1/consistency differ by class (is "contradictory" — the minority class — harder than the other two?),
- a couple of sentences on generalization limits beyond the four specified counterfactual families (genuine paraphrasing, second-language phrasing patterns),
- one sentence on production scaling: reference-embedding caching means marginal per-submission cost is one embedding pass, not two — the honest answer if a judge asks how this works beyond a hackathon-sized test set,
- an explicit line confirming no evaluation gold labels or the public corpus's answer key were loaded, queried, or memorized anywhere in the *training or feature* code path — naming the one script that's allowed to touch `test/` at all, and stating that its only output is a printed number.

## Reliability / reproducibility
- Pin dependency versions in `requirements.txt` from `pip freeze`, not guessed.
- Fix every random seed (numpy, python `random`, sklearn, torch if used).
- **Resolve the internet-access question as a go/no-go decision in the first 20 minutes, not a deferred note.** If using a pretrained embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`), the grading environment may have no internet access at eval time — actually test this (clear the local HF cache and try loading) rather than assuming vendoring "should" work. Either vendor the model weights into the submission and prove the fresh-clone test below succeeds with the cache cleared, or fall back to a pure TF-IDF + SVD similarity pipeline with zero download dependency. Pick one *now* and say why in the README — don't leave it as a live risk running into the final hour.
- One command (e.g. `python run_all.py`) that trains (or loads a saved model), evaluates, regenerates the counterfactuals, prints the scorecard, and writes the example explanations and bias write-up to disk. Test this from a genuinely fresh clone before calling it done — and if this is also the moment you test the internet-access assumption above, so much the better.

## Hard constraints — do not violate
- Never load, query, reconstruct, or fingerprint SemEval gold labels from the public corpus or any external copy, and never build a memorized answer→label lookup table. The model must generalize to unseen answers. **Concretely: `training/3way/**` may feed the model and CF4's vectorizer; `test/3way/**` may only be read by the one local-generalization script named in the bias write-up, and its only output is a printed number — nothing that feeds back into training, features, or thresholds.**
- The full pipeline (train + evaluate + counterfactual metrics) must run well inside 3 hours of wall-clock time. If the plan below runs long, cut stretch items first (demo UI, SHAP, the trailing-token heuristic) — never cut the robustness-by-design step, the guardrail check, or the reproducibility pass.

## Suggested time-boxed plan (turn this into your Planning-Mode checklist)
- 0:00–0:05 — confirm Antigravity's auto-execution policy and quota headroom (see "Before the clock starts")
- 0:05–0:25 — clone + inspect data against the verified layout above, build the XML loader, local stratified val split, sanity-check label balance against the measured 41/41/17 split
- 0:25–0:40 — implement CF1–CF4 exactly as above; hand-check on 3 toy examples including one with a contraction or hyphen
- 0:40–1:20 — canonicalization + embedding/feature pipeline (reference-embedding caching, multi-reference max-similarity, missing-reference fallback) + baseline classifier (Logistic Regression with `class_weight="balanced"` — not optional given the 17% minority class) + probability calibration
- 1:20–1:40 — implement the 4 metrics + guardrail check + scorecard script, run end to end
- 1:40–2:10 — robustness iteration using the scorecard's live numbers, watching the guardrail alongside consistency/CF4 the whole time (highest-leverage 30 minutes of the whole build)
- 2:10–2:30 — explainability outputs + bias/limitations write-up, including the real `test/3way` generalization numbers
- 2:30–2:50 — reproducibility pass: requirements.txt, seeds, README (internet-access decision + scaling sentence), fresh-clone test
- 2:50–3:00 — buffer, or stretch: minimal Streamlit/Gradio demo (verify it actually loads via the browser tool before calling it done)

## Deliverables checklist
- [ ] training/inference source code + environment file
- [ ] three-class probability output
- [ ] local counterfactual generator (CF1–CF4) that prints the robustness/calibration metrics on the original data, plus the guardrail check
- [ ] example evidence/explanations for a handful of predictions
- [ ] short limitations/bias analysis markdown file, including real `test/3way` generalization numbers by condition
- [ ] README with exact run instructions, stated assumptions (including the 3-hour-vs-5–7-hour framing and the internet-access decision), and expected runtime

Suggested repo layout:
```
ed05-submission/
  data/                  # cloned SemEval repo, or a fetch script — don't commit huge raw data
  src/
    load_data.py          # training/3way/** only
    preprocess.py         # canonicalization + boilerplate stripping live here
    counterfactuals.py    # CF1-CF4
    features.py
    model.py
    metrics.py             # includes the guardrail check
    explain.py
  local_eval.py           # the ONLY script allowed to touch test/3way/** — prints numbers, feeds nothing back
  run_all.py
  requirements.txt
  README.md
  BIAS_ANALYSIS.md
  app.py                  # optional stretch demo
```
