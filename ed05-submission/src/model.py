"""
Model training and inference for ED-05 short-answer grading.
Logistic Regression with balanced class weights + probability calibration.
"""

import numpy as np
import pickle
import random
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features import init_models, cache_reference_embeddings, extract_features_batch
from src.preprocess import preprocess
from src.metrics import CLASS_ORDER

# Fix random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

MODEL_DIR = Path(__file__).parent.parent / "models"


def build_classifier():
    """Build the classification pipeline: StandardScaler + Logistic Regression."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs',
        )),
    ])
    return pipeline


def train_model(train_data: list, val_data: list = None, use_nli: bool = False,
                calibrate: bool = True):
    """Train the classifier on training data.

    Args:
        train_data: list of dicts from load_data
        val_data: optional validation data for calibration
        use_nli: whether to use NLI cross-encoder features
        calibrate: whether to apply probability calibration

    Returns:
        Trained pipeline (with calibration if requested)
    """
    print("\n" + "=" * 50)
    print("  Training Model")
    print("=" * 50)

    # Initialize models
    init_models(use_nli=use_nli)

    # Cache reference embeddings
    all_data = train_data + (val_data if val_data else [])
    cache_reference_embeddings(all_data)

    # Extract features
    print("  Extracting training features...")
    X_train = extract_features_batch(train_data, use_nli=use_nli)
    y_train = np.array([d['label'] for d in train_data])
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Build and train base classifier
    pipeline = build_classifier()

    if calibrate and val_data:
        print("  Extracting validation features for calibration...")
        X_val = extract_features_batch(val_data, use_nli=use_nli)
        y_val = np.array([d['label'] for d in val_data])
        print(f"  Validation set: {X_val.shape[0]} samples")

        # Train base pipeline first
        pipeline.fit(X_train, y_train)
        print(f"  Base model accuracy: {pipeline.score(X_val, y_val):.4f}")

        # Calibrate with CalibratedClassifierCV using sigmoid (Platt scaling)
        # Fit on the full training set with cross-validation
        print("  Calibrating probabilities (Platt scaling, 5-fold CV)...")
        calibrated = CalibratedClassifierCV(
            pipeline,
            method='sigmoid',
            cv=5,
        )
        calibrated.fit(X_train, y_train)
        print(f"  Calibrated model accuracy: {calibrated.score(X_val, y_val):.4f}")
        model = calibrated
    else:
        pipeline.fit(X_train, y_train)
        model = pipeline

    return model


def predict(model, data: list, use_nli: bool = False):
    """Make predictions on data.

    Returns:
        preds: list of predicted class labels
        probs: array of shape (N, 3) with probabilities for CLASS_ORDER
    """
    X = extract_features_batch(data, use_nli=use_nli)
    preds = model.predict(X).tolist()
    probs = model.predict_proba(X)

    # Ensure probs columns match CLASS_ORDER
    if hasattr(model, 'classes_'):
        model_classes = list(model.classes_)
    else:
        # CalibratedClassifierCV
        model_classes = list(model.classes_)

    # Reorder probs to match CLASS_ORDER if needed
    if list(model_classes) != CLASS_ORDER:
        reordered = np.zeros_like(probs)
        for i, cls in enumerate(CLASS_ORDER):
            if cls in model_classes:
                src_idx = model_classes.index(cls)
                reordered[:, i] = probs[:, src_idx]
        probs = reordered

    return preds, probs


def predict_single(model, record: dict, use_nli: bool = False):
    """Predict for a single record. Returns (pred_label, prob_vector)."""
    preds, probs = predict(model, [record], use_nli=use_nli)
    return preds[0], probs[0]


def save_model(model, filename='model.pkl'):
    """Save trained model to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / filename
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved to {path}")


def load_model(filename='model.pkl'):
    """Load trained model from disk."""
    path = MODEL_DIR / filename
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"  Model loaded from {path}")
    return model


if __name__ == '__main__':
    print("Model module loaded. Use train_model() in run_all.py.")
