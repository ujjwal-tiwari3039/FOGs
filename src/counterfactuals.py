import numpy as np
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logger = logging.getLogger(__name__)

class CounterfactualGenerator:
    def __init__(self, corpus=None):
        """
        Initializes the generator. CF4 requires a corpus to fit TF-IDF.
        """
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        if corpus is not None:
            self.fit_tfidf(corpus)

    def fit_tfidf(self, corpus):
        """
        Fits the TfidfVectorizer on the provided corpus.
        """
        logger.info("Fitting TF-IDF vectorizer for CF4...")
        self.vectorizer.fit(corpus)

    def generate(self, answer):
        """
        Generates CF1, CF2, CF3, CF4 for a given answer.
        """
        if not answer:
            answer = ""

        # CF1: Lowercase and remove punctuation
        cf1 = answer.lower().translate(str.maketrans('', '', string.punctuation))

        # CF2: Prepend "In my answer, I think that"
        cf2 = f"In my answer, I think that {answer}"

        # CF3: Append "This is my final answer."
        cf3 = f"{answer} This is my final answer."

        # CF4: Append top 2 TF-IDF tokens twice
        cf4 = self._generate_cf4(answer)

        return {
            "original": answer,
            "cf1": cf1,
            "cf2": cf2,
            "cf3": cf3,
            "cf4": cf4
        }

    def _generate_cf4(self, answer):
        """
        Implements the exact CF4 procedure:
        1. Identify two eligible tokens with largest TF-IDF.
        2. Break ties lexicographically.
        3. Append those two tokens twice.
        """
        try:
            tfidf_matrix = self.vectorizer.transform([answer])
            feature_names = self.vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Find indices of non-zero scores
            eligible_indices = np.where(tfidf_scores > 0)[0]

            if len(eligible_indices) == 0:
                return answer

            # Get scores and names for eligible tokens
            eligible_scores = tfidf_scores[eligible_indices]
            eligible_names = feature_names[eligible_indices]

            # Sort by score (descending) then by name (ascending) for tie-breaking
            # We use a stable sort or a custom key.
            # Sort names first, then scores.
            sorted_indices = np.lexsort((eligible_names, -eligible_scores))
            top_tokens = eligible_names[sorted_indices[:2]]

            # Append these tokens twice
            suffix = " ".join([t * 2 for t in top_tokens]) # This is wrong, "token token" not "tokentoken"
            # The prompt says "Append those two tokens twice" -> "token token token token" or "token token" ?
            # "Append those two tokens twice" usually means repeat each token twice.
            # Let's use: " token token token token"
            suffix = " " + " ".join([f"{t} {t}" for t in top_tokens])
            return answer + suffix

        except Exception as e:
            logger.error(f"CF4 generation failed: {e}")
            return answer
