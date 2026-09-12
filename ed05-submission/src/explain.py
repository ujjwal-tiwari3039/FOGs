"""
Explainability outputs for ED-05: evidence strings, confidence flags,
and example prediction explanations.
"""

import numpy as np
from src.preprocess import preprocess, canonicalize
from src.metrics import CLASS_ORDER

# Confidence threshold: top-class probability below this → "low confidence"
CONFIDENCE_THRESHOLD = 0.50
# Margin threshold: difference between top-2 class probabilities
MARGIN_THRESHOLD = 0.15

# Simple negation cues for heuristic detection
NEGATION_CUES = [
    'not', 'no', 'never', 'neither', 'nor', "n't", 'dont', 'doesnt',
    'didnt', 'isnt', 'wasnt', 'werent', 'wont', 'cant', 'cannot',
    'without', 'nothing', 'nowhere', 'none',
]


def _get_content_words(text: str) -> set:
    """Get content words from preprocessed text (non-stopword, len > 2)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words = canonicalize(text).split()
    return {w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2}


def confidence_flag(prob_vector: np.ndarray) -> dict:
    """Assess prediction confidence.

    Returns dict with:
        - 'confident': bool
        - 'reason': str (if not confident)
        - 'top_prob': float
        - 'margin': float (gap between top-2 classes)
    """
    sorted_probs = np.sort(prob_vector)[::-1]
    top_prob = sorted_probs[0]
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

    result = {
        'top_prob': float(top_prob),
        'margin': float(margin),
        'confident': True,
        'reason': '',
    }

    if top_prob < CONFIDENCE_THRESHOLD:
        result['confident'] = False
        result['reason'] = f"Top-class probability ({top_prob:.3f}) below threshold ({CONFIDENCE_THRESHOLD})"
    elif margin < MARGIN_THRESHOLD:
        result['confident'] = False
        result['reason'] = f"Thin margin ({margin:.3f}) between top-2 classes"

    return result


def detect_negation(student_text: str, reference_text: str) -> dict:
    """Simple negation/contradiction cue detection.

    Returns dict with:
        - 'has_negation_cue': bool
        - 'cues_found': list of negation words in student answer
        - 'note': str
    """
    student_words = set(canonicalize(student_text).split())
    ref_words = set(canonicalize(reference_text).split())

    student_neg = [w for w in NEGATION_CUES if w in student_words]
    ref_neg = [w for w in NEGATION_CUES if w in ref_words]

    # Negation mismatch: student has negation but ref doesn't, or vice versa
    has_mismatch = bool(student_neg) != bool(ref_neg)

    return {
        'has_negation_cue': has_mismatch or bool(student_neg),
        'cues_found': student_neg,
        'note': ("possible negation — student answer contains negation cue(s) "
                 "not present in reference" if has_mismatch
                 else ("negation cue detected in both student and reference"
                       if student_neg and ref_neg
                       else ""))
    }


def generate_evidence(record: dict, pred_label: str, prob_vector: np.ndarray) -> dict:
    """Generate explainability evidence for a single prediction.

    Returns a dict with structured evidence for human review.
    """
    student_text = record['student_answer']
    ref_texts = record.get('reference_answers', [])
    question_text = record['question_text']

    student_words = _get_content_words(student_text)

    # Find shared and missing words across all references
    all_ref_words = set()
    for ref in ref_texts:
        all_ref_words.update(_get_content_words(ref))

    shared_words = student_words & all_ref_words
    missing_from_student = all_ref_words - student_words
    extra_in_student = student_words - all_ref_words

    # Negation detection (against best reference if available)
    neg_result = {'has_negation_cue': False, 'cues_found': [], 'note': ''}
    if ref_texts:
        # Check against each reference, flag if any shows mismatch
        for ref in ref_texts:
            neg = detect_negation(student_text, ref)
            if neg['has_negation_cue']:
                neg_result = neg
                break

    # Confidence assessment
    conf = confidence_flag(prob_vector)

    # Build evidence string
    evidence_parts = []
    if shared_words:
        evidence_parts.append(f"Shared content words: {', '.join(sorted(shared_words)[:10])}")
    if missing_from_student:
        evidence_parts.append(f"Key reference words missing: {', '.join(sorted(missing_from_student)[:8])}")
    if extra_in_student:
        evidence_parts.append(f"Extra words in student answer: {', '.join(sorted(extra_in_student)[:5])}")
    if neg_result['has_negation_cue']:
        evidence_parts.append(f"⚠ {neg_result['note']} (cues: {neg_result['cues_found']})")

    return {
        'question': question_text,
        'student_answer': student_text,
        'reference_answers': ref_texts,
        'predicted_class': pred_label,
        'probability_vector': {cls: float(prob_vector[i]) for i, cls in enumerate(CLASS_ORDER)},
        'confidence': conf,
        'evidence_string': ' | '.join(evidence_parts) if evidence_parts else "No notable evidence",
        'shared_words': sorted(shared_words),
        'missing_words': sorted(missing_from_student),
        'negation': neg_result,
    }


def format_explanation(evidence: dict, idx: int = 0) -> str:
    """Format a single prediction explanation as a readable string."""
    lines = []
    lines.append(f"\n{'─'*60}")
    lines.append(f"  Example #{idx + 1}")
    lines.append(f"{'─'*60}")
    lines.append(f"  Question:  {evidence['question'][:100]}...")
    lines.append(f"  Student:   {evidence['student_answer']}")
    if evidence['reference_answers']:
        lines.append(f"  Reference: {evidence['reference_answers'][0]}")
        if len(evidence['reference_answers']) > 1:
            lines.append(f"             (+{len(evidence['reference_answers'])-1} more)")

    lines.append(f"\n  Predicted: {evidence['predicted_class'].upper()}")
    probs = evidence['probability_vector']
    lines.append(f"  Probabilities: " + " | ".join(
        f"{cls}: {p:.3f}" for cls, p in probs.items()
    ))

    conf = evidence['confidence']
    if conf['confident']:
        lines.append(f"  Confidence: ✅ HIGH (top={conf['top_prob']:.3f}, margin={conf['margin']:.3f})")
    else:
        lines.append(f"  Confidence: ⚠️  LOW — RECOMMEND HUMAN REVIEW")
        lines.append(f"    Reason: {conf['reason']}")

    lines.append(f"\n  Evidence: {evidence['evidence_string']}")

    if evidence['negation']['has_negation_cue']:
        lines.append(f"  Negation: ⚠ {evidence['negation']['note']}")

    return '\n'.join(lines)


def generate_example_explanations(data: list, preds: list, probs: np.ndarray,
                                  n_examples: int = 10) -> str:
    """Generate explanations for a representative sample of predictions.

    Selects examples spread across all three classes, plus at least one
    CF-related pair if available.
    """
    output_lines = []
    output_lines.append("\n" + "=" * 60)
    output_lines.append("  EXAMPLE PREDICTIONS WITH EXPLANATIONS")
    output_lines.append("=" * 60)

    # Select examples: ~3-4 per class, prioritize diverse confidence levels
    selected_indices = []
    for target_class in CLASS_ORDER:
        class_indices = [i for i, p in enumerate(preds) if p == target_class]
        if not class_indices:
            continue

        # Get confidence scores
        class_confs = [(i, probs[i][CLASS_ORDER.index(target_class)]) for i in class_indices]
        class_confs.sort(key=lambda x: x[1])

        # Pick high-confidence, low-confidence, and mid
        picks = []
        if len(class_confs) >= 3:
            picks = [class_confs[-1][0], class_confs[0][0], class_confs[len(class_confs)//2][0]]
        else:
            picks = [c[0] for c in class_confs[:3]]

        selected_indices.extend(picks[:min(4, n_examples // 3 + 1)])

    # Deduplicate and limit
    selected_indices = list(dict.fromkeys(selected_indices))[:n_examples]

    for idx_num, data_idx in enumerate(selected_indices):
        evidence = generate_evidence(data[data_idx], preds[data_idx], probs[data_idx])
        output_lines.append(format_explanation(evidence, idx_num))

    output_lines.append("\n" + "=" * 60)

    # Summary stats
    confident_count = sum(
        1 for i in range(len(preds))
        if confidence_flag(probs[i])['confident']
    )
    output_lines.append(
        f"  Overall: {confident_count}/{len(preds)} predictions are high-confidence "
        f"({100*confident_count/len(preds):.1f}%)"
    )
    low_conf_count = len(preds) - confident_count
    output_lines.append(
        f"  {low_conf_count} predictions flagged for human review"
    )

    return '\n'.join(output_lines)


if __name__ == '__main__':
    print("Explain module loaded. Use generate_example_explanations() in run_all.py.")
