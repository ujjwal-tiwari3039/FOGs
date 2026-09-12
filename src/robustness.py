import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def calculate_robustness(y_true, y_prob_orig, y_prob_cfs):
    """
    Calculates robustness metrics.

    Args:
        y_true: (N,) ground truth labels
        y_prob_orig: (N, 3) probabilities for original
        y_prob_cfs: dict { "cf1": (N, 3), ... }

    Returns:
        metrics: dict of robustness results
    """
    # Label mapping to index for 'correct'
    # Assume classes are ['correct', 'contradictory', 'incorrect']
    # We need to find the index of 'correct'
    # Since we'll use a fixed order in the pipeline, let's assume 0 is 'correct'
    # But for robustness, we should be explicit.
    correct_idx = 0

    results = {}

    for cf_name, y_prob_cf in y_prob_cfs.items():
        # 1. Prediction consistency
        pred_orig = np.argmax(y_prob_orig, axis=1)
        pred_cf = np.argmax(y_prob_cf, axis=1)
        consistency = np.mean(pred_orig == pred_cf)

        # 2. Average absolute probability difference
        abs_diff = np.mean(np.abs(y_prob_orig - y_prob_cf))

        # 3. Correct-score inflation (for CF4 specifically)
        inflation = np.mean(y_prob_cf[:, correct_idx] - y_prob_orig[:, correct_idx])

        results[cf_name] = {
            "consistency": consistency,
            "avg_abs_diff": abs_diff,
            "inflation": inflation,
            "max_inflation": np.max(y_prob_cf[:, correct_idx] - y_prob_orig[:, correct_idx]),
            "median_inflation": np.median(y_prob_cf[:, correct_idx] - y_prob_orig[:, correct_idx])
        }

    return results
