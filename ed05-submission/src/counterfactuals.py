"""
Counterfactual generators CF1-CF4 for ED-05 robustness evaluation.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def cf1(text: str) -> str:
    """CF1: lowercase + strip punctuation via [^\\w\\s] removal.
    Same transformation as canonicalize() in preprocess.py.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def cf2(text: str) -> str:
    """CF2: prepend 'In my answer, I think that ' to the original text."""
    return "In my answer, I think that " + text


def cf3(text: str) -> str:
    """CF3: append ' This is my final answer.' to the original text."""
    return text + " This is my final answer."


def fit_cf4_vectorizer(training_answers: list) -> TfidfVectorizer:
    """Fit TF-IDF vectorizer on training student answers for CF4.

    Uses sklearn defaults: lowercase=True, stop_words='english',
    default token_pattern (not overridden).
    """
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    vectorizer.fit(training_answers)
    return vectorizer


def cf4(text: str, vectorizer: TfidfVectorizer) -> str:
    """CF4: append top-2 TF-IDF tokens from the answer.

    Tokens are sorted by descending TF-IDF weight, ties broken alphabetically
    ascending. Top 2 are appended space-separated at the end.
    """
    tfidf_matrix = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Get nonzero weights for this text
    nonzero_indices = tfidf_matrix.nonzero()[1]
    if len(nonzero_indices) == 0:
        return text

    weights = tfidf_matrix.toarray()[0]
    eligible_tokens = []
    for idx in nonzero_indices:
        eligible_tokens.append((feature_names[idx], weights[idx]))

    # Sort by descending weight, then alphabetically ascending for ties
    eligible_tokens.sort(key=lambda x: (-x[1], x[0]))

    # Take top 2
    top_tokens = [t[0] for t in eligible_tokens[:2]]

    return text + " " + " ".join(top_tokens)


def generate_all_counterfactuals(text: str, vectorizer: TfidfVectorizer) -> dict:
    """Generate all 4 counterfactual variants of a text."""
    return {
        'original': text,
        'cf1': cf1(text),
        'cf2': cf2(text),
        'cf3': cf3(text),
        'cf4': cf4(text, vectorizer),
    }


if __name__ == '__main__':
    print("=" * 60)
    print("  Counterfactual Generator Tests")
    print("=" * 60)

    # Example training answers for fitting vectorizer
    training_examples = [
        "The water evaporated, leaving the salt behind.",
        "Terminal 1 and the positive terminal are separated by the gap.",
        "The bulbs will stay lit because the circuit is closed.",
        "Energy is transferred from the battery to the bulb.",
        "The switch opens the circuit, breaking the connection.",
        "Current flows through the wire from positive to negative.",
        "The voltage reading shows a difference in electrical potential.",
        "Evaporation is the process of water turning into vapor.",
    ]

    vectorizer = fit_cf4_vectorizer(training_examples)
    print(f"  Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")

    test_cases = [
        "The water evaporated.",
        "Don't touch the terminal!",
        "The single-pole switch is open.",
    ]

    for text in test_cases:
        print(f"\n  Original: {text!r}")
        cfs = generate_all_counterfactuals(text, vectorizer)
        for name, val in cfs.items():
            if name != 'original':
                print(f"    {name}: {val!r}")
