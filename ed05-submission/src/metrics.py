"""
Evaluation metrics for ED-05, matching the rubric exactly.
Includes guardrail checks for degenerate classifiers.
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report
from collections import Counter

CLASS_ORDER = ['correct', 'contradictory', 'incorrect']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 across the three classes."""
    return f1_score(y_true, y_pred, average='macro', labels=CLASS_ORDER)


def counterfactual_consistency(original_preds: list, cf_preds: list) -> float:
    """Consistency of CF1+CF2+CF3 predictions with originals.

    Args:
        original_preds: predicted class for each original input
        cf_preds: list of lists; cf_preds[i] = [cf1_pred, cf2_pred, cf3_pred] for example i

    Returns:
        Fraction of all CF predictions matching the corresponding original.
    """
    total = 0
    matches = 0
    for orig, cfs in zip(original_preds, cf_preds):
        for cf_pred in cfs:
            total += 1
            if cf_pred == orig:
                matches += 1
    return matches / total if total > 0 else 1.0


def cf4_resistance(p_correct_original: list, p_correct_cf4: list) -> float:
    """CF4 keyword-stuffing resistance utility.

    U_CF4 = 1 - mean(max(0, P_correct(CF4_i) - P_correct(original_i)))
    Clipped to [0, 1].
    """
    diffs = []
    for p_orig, p_cf4 in zip(p_correct_original, p_correct_cf4):
        diffs.append(max(0.0, p_cf4 - p_orig))
    u = 1.0 - np.mean(diffs)
    return float(np.clip(u, 0.0, 1.0))


def normalized_brier_score(y_true_labels, p_probs) -> float:
    """Normalized multiclass Brier score.

    Formula: (1/(2N)) * sum_i sum_k (p_ik - y_ik)^2
    Note: 1/(2N), NOT 1/N. This keeps the score in [0, 1].

    Args:
        y_true_labels: list of true label strings
        p_probs: array of shape (N, 3) - probabilities for [correct, contradictory, incorrect]
    """
    n = len(y_true_labels)
    if n == 0:
        return 0.0

    p_probs = np.array(p_probs)

    # One-hot encode true labels
    y_onehot = np.zeros((n, 3))
    for i, label in enumerate(y_true_labels):
        y_onehot[i, CLASS_TO_IDX[label]] = 1.0

    brier = np.sum((p_probs - y_onehot) ** 2) / (2 * n)
    return float(brier)


def guardrail_check(y_pred, y_true=None) -> dict:
    """Check for degenerate classifier behavior.

    Returns dict with 'distribution', 'recalls' (if y_true given),
    'warnings' (list of warning strings).
    """
    result = {'distribution': {}, 'warnings': []}

    # Predicted class distribution
    pred_counts = Counter(y_pred)
    total = len(y_pred)
    for cls in CLASS_ORDER:
        count = pred_counts.get(cls, 0)
        pct = 100 * count / total if total > 0 else 0
        result['distribution'][cls] = {'count': count, 'pct': pct}

    # Check for collapsed distribution
    active_classes = sum(1 for cls in CLASS_ORDER if pred_counts.get(cls, 0) > 0)
    if active_classes <= 2:
        result['warnings'].append(
            f"⚠️  DEGENERATE: Only {active_classes} classes predicted! "
            f"Distribution: {dict(pred_counts)}"
        )

    # Check for very low prediction rates
    for cls in CLASS_ORDER:
        pct = result['distribution'][cls]['pct']
        if 0 < pct < 5:
            result['warnings'].append(
                f"⚠️  Class '{cls}' has only {pct:.1f}% of predictions"
            )

    # Per-class recall if ground truth provided
    if y_true is not None:
        result['recalls'] = {}
        for cls in CLASS_ORDER:
            true_count = sum(1 for y in y_true if y == cls)
            if true_count == 0:
                result['recalls'][cls] = float('nan')
                continue
            correct = sum(1 for yt, yp in zip(y_true, y_pred)
                          if yt == cls and yp == cls)
            recall = correct / true_count
            result['recalls'][cls] = recall
            if recall == 0:
                result['warnings'].append(
                    f"⚠️  ZERO RECALL for class '{cls}'! "
                    f"({true_count} true examples, none predicted correctly)"
                )

    return result


def print_scorecard(f1, consistency, cf4_utility, brier, guardrail_result,
                    reproducibility_score=5.0):
    """Print formatted scorecard matching the ED-05 rubric."""
    cal_score = 1.0 - brier
    total = (f1 * 45 + consistency * 30 + cf4_utility * 15 +
             cal_score * 5 + reproducibility_score)

    print("\n" + "═" * 50)
    print("  ED-05 SCORECARD")
    print("═" * 50)
    print(f"  Macro-F1:           {f1:.4f}  → {f1*45:.1f}/45")
    print(f"  CF Consistency:     {consistency:.4f}  → {consistency*30:.1f}/30")
    print(f"  CF4 Resistance:     {cf4_utility:.4f}  → {cf4_utility*15:.1f}/15")
    print(f"  Calibration (1-B):  {cal_score:.4f}  → {cal_score*5:.1f}/5")
    print(f"  Reproducibility:    {reproducibility_score:.1f}/5")
    print("─" * 50)
    print(f"  TOTAL:              {total:.1f}/100")
    print("═" * 50)

    # Guardrail output
    if guardrail_result['warnings']:
        print("\n  ⚠️  GUARDRAIL WARNINGS:")
        for w in guardrail_result['warnings']:
            print(f"    {w}")

    print("\n  Predicted class distribution:")
    for cls, info in guardrail_result['distribution'].items():
        print(f"    {cls:15s}: {info['count']:5d}  ({info['pct']:5.1f}%)")

    if 'recalls' in guardrail_result:
        print("\n  Per-class recall:")
        for cls, recall in guardrail_result['recalls'].items():
            if np.isnan(recall):
                print(f"    {cls:15s}: N/A (no true examples)")
            else:
                print(f"    {cls:15s}: {recall:.4f}")

    print()
    return total


if __name__ == '__main__':
    print("Metrics module self-test")
    print("=" * 50)

    # Synthetic test
    y_true = ['correct', 'correct', 'incorrect', 'incorrect', 'contradictory',
              'correct', 'incorrect', 'contradictory', 'correct', 'incorrect']
    y_pred = ['correct', 'incorrect', 'incorrect', 'incorrect', 'contradictory',
              'correct', 'correct', 'incorrect', 'correct', 'incorrect']

    print(f"Macro-F1: {macro_f1(y_true, y_pred):.4f}")

    # CF consistency
    orig_preds = ['correct', 'incorrect', 'contradictory']
    cf_preds_list = [
        ['correct', 'correct', 'correct'],       # all match
        ['incorrect', 'incorrect', 'correct'],    # 2 match
        ['contradictory', 'incorrect', 'correct'],  # 1 matches
    ]
    print(f"CF Consistency: {counterfactual_consistency(orig_preds, cf_preds_list):.4f}")

    # CF4 resistance
    p_orig = [0.8, 0.3, 0.1]
    p_cf4 = [0.9, 0.4, 0.2]
    print(f"CF4 Resistance: {cf4_resistance(p_orig, p_cf4):.4f}")

    # Brier
    probs = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
        [0.1, 0.8, 0.1],
    ])
    labels = ['correct', 'incorrect', 'contradictory']
    print(f"Brier Score: {normalized_brier_score(labels, probs):.4f}")

    # Guardrail
    g = guardrail_check(y_pred, y_true)
    print(f"Guardrail warnings: {g['warnings']}")
