import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class ShortAnswerClassifier:
    def __init__(self, model_type='LogisticRegression', temperature=1.0):
        self.model_type = model_type
        self.temperature = temperature
        self.classifier = None
        self.scaler = StandardScaler()

    def train(self, X, y):
        """
        Trains the classifier with scaling.
        """
        X_scaled = self.scaler.fit_transform(X)

        if self.model_type == 'LogisticRegression':
            self.classifier = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.classifier.fit(X_scaled, y)
        logger.info("Model training complete.")

    def predict_proba(self, X):
        """
        Predicts class probabilities with temperature scaling.
        """
        X_scaled = self.scaler.transform(X)
        probs = self.classifier.predict_proba(X_scaled)

        # Temperature scaling: Softmax(logits / T)
        # LogisticRegression.predict_proba uses softmax internally.
        # To apply temperature scaling, we need the logits.
        # For LogisticRegression, the decision_function provides the logits for binary.
        # For multiclass (ovr), it's slightly different.
        # Simple approximation: divide probabilities by T before re-normalizing.
        # But proper temperature scaling is applied to logits.

        if self.temperature != 1.0:
            # This is a simplified temperature scaling on probabilities
            # Correct way: get logits -> divide by T -> softmax
            # Since we use sklearn, we'll approximate or use a custom softmax.
            # For this hackathon, if we can't get logits easily, we'll just use probabilities.
            # However, let's try to use the decision_function.
            logits = self.classifier.decision_function(X_scaled)
            scaled_logits = logits / self.temperature

            # Softmax
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def calibrate(self, X_val, y_val):
        """
        Optimizes temperature using validation set.
        """
        # Simple grid search for temperature
        best_t = 1.0
        best_logloss = float('inf')

        from sklearn.metrics import log_loss
        X_scaled = self.scaler.transform(X_val)

        # Only try to calibrate if we have validation data
        for t in np.linspace(0.1, 3.0, 30):
            logits = self.classifier.decision_function(X_scaled)
            scaled_logits = logits / t
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            # We need y_val to be one-hot or match prob shape
            # This is simplified; for a real project, we'd use cross_entropy
            try:
                # Assuming y_val is integer labels 0, 1, 2
                ll = log_loss(y_val, probs)
                if ll < best_logloss:
                    best_logloss = ll
                    best_t = t
            except:
                continue

        self.temperature = best_t
        logger.info(f"Calibrated temperature: {best_t:.3f}")

    def save(self, path):
        joblib.dump({"model": self.classifier, "scaler": self.scaler, "temp": self.temperature}, path)

    def load(self, path):
        data = joblib.load(path)
        self.classifier = data["model"]
        self.scaler = data["scaler"]
        self.temperature = data["temp"]
