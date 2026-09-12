import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
import logging

logger = logging.getLogger(__name__)

def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculates the Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_low = bin_boundaries[i]
        bin_high = bin_boundaries[i+1]

        # Filter samples in this bin (using the probability of the predicted class)
        # For multiclass, ECE is usually calculated per class or on the max prob
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)

        mask = (confidences >= bin_low) & (confidences < bin_high)
        if np.any(mask):
            bin_acc = np.mean(predictions[mask] == y_true[mask])
            bin_conf = np.mean(confidences[mask])
            ece += np.mean(mask) * np.abs(bin_acc - bin_conf)

    return ece

def get_calibration_metrics(y_true, y_prob):
    """
    Returns a dictionary of calibration metrics.
    """
    # For Brier score, we need one-hot encoding
    # Brier score is usually defined for binary; for multiclass, it's the sum of squares
    # over all classes.

    # Convert y_true to one-hot
    n_classes = y_prob.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]

    brier = np.mean(np.sum((y_true_onehot - y_prob)**2, axis=1))
    ll = log_loss(y_true, y_prob)
    ece = calculate_ece(y_true, y_prob)

    return {
        "log_loss": ll,
        "brier_score": brier,
        "ece": ece
    }
