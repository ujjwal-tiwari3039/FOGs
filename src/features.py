import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, include_lexical=False):
        self.include_lexical = include_lexical

    def extract(self, student_embeddings, reference_embeddings, student_texts=None, reference_texts=None):
        """
        Extracts semantic features from embeddings.
        """
        # 1. Cosine Similarity
        norm_s = np.linalg.norm(student_embeddings, axis=1)
        norm_r = np.linalg.norm(reference_embeddings, axis=1)
        cos_sim = (student_embeddings * reference_embeddings).sum(axis=1) / (norm_s * norm_r + 1e-9)

        # 2. Absolute Difference
        abs_diff = np.linalg.norm(student_embeddings - reference_embeddings, axis=1)

        # 3. Element-wise Product (mean)
        prod_mean = np.mean(student_embeddings * reference_embeddings, axis=1)

        # 4. Length difference
        length_diff = 0
        if student_texts is not None and reference_texts is not None:
            s_lens = np.array([len(t.split()) for t in student_texts])
            r_lens = np.array([len(t.split()) for t in reference_texts])
            length_diff = np.abs(s_lens - r_lens)

        semantic_features = np.column_stack([cos_sim, abs_diff, prod_mean, length_diff])
        combined_embeddings = np.hstack([student_embeddings, reference_embeddings])

        features = np.hstack([semantic_features, combined_embeddings])

        if self.include_lexical and student_texts is not None and reference_texts is not None:
            lexical_features = []
            for s, r in zip(student_texts, reference_texts):
                s_set = set(s.lower().split())
                r_set = set(r.lower().split())
                overlap = len(s_set & r_set) / (len(r_set) + 1e-9)
                lexical_features.append(overlap)
            features = np.hstack([features, np.array(lexical_features).reshape(-1, 1)])

        return features
