import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_prob):
    """
    Comprehensive evaluation of the classifier.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # Per-class metrics
    # labels = ['correct', 'contradictory', 'incorrect']
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "per_class": {
            "correct": {"precision": precision[0], "recall": recall[0], "f1": f1[0]},
            "contradictory": {"precision": precision[1], "recall": recall[1], "f1": f1[1]},
            "incorrect": {"precision": precision[2], "recall": recall[2], "f1": f1[2]},
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    return metrics
