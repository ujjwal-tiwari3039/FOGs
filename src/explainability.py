import numpy as np
import logging

logger = logging.getLogger(__name__)

class Explainer:
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor

    def explain(self, student_text, reference_text, student_emb, reference_emb):
        """
        Provides a human-readable explanation for the prediction.
        """
        # Extract features for this single example
        feats = self.feature_extractor.extract(
            student_emb.reshape(1, -1),
            reference_emb.reshape(1, -1),
            [student_text],
            [reference_text]
        )

        # Get probabilities
        probs = self.model.predict_proba(feats)[0]
        pred_class_idx = np.argmax(probs)

        # Map index to label
        labels = ['correct', 'contradictory', 'incorrect']
        pred_label = labels[pred_class_idx]

        # Semantic Similarity (the first feature)
        sim = feats[0, 0]

        evidence = []
        if pred_label == 'correct':
            evidence.append(f"Student response is semantically close to the reference answer (Similarity: {sim:.2f}).")
            if sim > 0.8:
                evidence.append("The confidence is high because semantic similarity is very strong.")
            else:
                evidence.append("The response captures the core meaning despite some stylistic differences.")

        elif pred_label == 'contradictory':
            evidence.append(f"The response contains information that conflicts with the reference answer.")
            evidence.append(f"Semantic similarity is low or mid ({sim:.2f}), but the model detects a contradictory pattern.")

        else: # incorrect
            evidence.append(f"The response fails to contain the required core concept (Similarity: {sim:.2f}).")
            evidence.append("The answer is either too vague or entirely off-topic.")

        return {
            "prediction": pred_label.upper(),
            "evidence": evidence,
            "similarity": sim,
            "probabilities": {labels[i]: float(probs[i]) for i in range(3)}
        }
