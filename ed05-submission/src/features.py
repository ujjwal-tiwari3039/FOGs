"""
Feature extraction for ED-05: sentence-embedding cosine similarity,
optional NLI cross-encoder features, and reference-embedding caching.
"""

import numpy as np
import hashlib
from src.preprocess import preprocess

# Will be populated by init_models()
_sentence_model = None
_nli_model = None
_ref_embedding_cache = {}  # question_id -> list of embeddings

CLASS_ORDER = ['correct', 'contradictory', 'incorrect']


def init_models(use_nli=False):
    """Initialize sentence-transformer (and optionally NLI cross-encoder).

    Models are loaded once and cached as module-level globals.
    """
    global _sentence_model, _nli_model

    from sentence_transformers import SentenceTransformer

    if _sentence_model is None:
        print("  Loading sentence-transformers model...")
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  Sentence model loaded.")

    if use_nli and _nli_model is None:
        try:
            from sentence_transformers import CrossEncoder
            print("  Loading NLI cross-encoder...")
            _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-xsmall')
            print("  NLI model loaded.")
        except Exception as e:
            print(f"  WARNING: Could not load NLI model: {e}")
            _nli_model = None


def _get_embedding(text: str) -> np.ndarray:
    """Get sentence embedding for preprocessed text."""
    return _sentence_model.encode(text, convert_to_numpy=True)


def _get_embeddings_batch(texts: list) -> np.ndarray:
    """Get sentence embeddings for a batch of texts."""
    return _sentence_model.encode(texts, convert_to_numpy=True, batch_size=128)


def cache_reference_embeddings(data: list):
    """Pre-compute and cache reference answer embeddings per question.

    Each question's references are embedded once and stored by question_id.
    This is called once during training/setup, not per student answer.
    """
    global _ref_embedding_cache

    # Collect unique question_id -> reference_answers mapping
    q_refs = {}
    for d in data:
        qid = d['question_id']
        if qid not in q_refs:
            q_refs[qid] = d['reference_answers']

    print(f"  Caching embeddings for {len(q_refs)} questions...")

    for qid, refs in q_refs.items():
        if qid in _ref_embedding_cache:
            continue
        if refs:
            # Preprocess each reference answer, then embed
            preprocessed_refs = [preprocess(r) for r in refs]
            embeddings = _get_embeddings_batch(preprocessed_refs)
            _ref_embedding_cache[qid] = embeddings
        else:
            _ref_embedding_cache[qid] = None

    print(f"  Cached {len(_ref_embedding_cache)} question reference embeddings.")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def extract_features(record: dict, use_nli: bool = False) -> np.ndarray:
    """Extract feature vector for a single student answer.

    Features:
    1. Max cosine similarity across all reference answers (or 0 if no refs)
    2. Mean cosine similarity across all reference answers (or 0 if no refs)
    3. Student answer length (word count, log-scaled)
    4. Reference answer length (avg, log-scaled)
    5. Length ratio (student / avg reference)
    6-8. NLI scores [contradiction, entailment, neutral] if use_nli=True, else 0

    Returns:
        Feature vector of shape (n_features,)
    """
    student_text = preprocess(record['student_answer'])
    question_text = preprocess(record['question_text'])
    qid = record['question_id']

    student_emb = _get_embedding(student_text)
    question_emb = _get_embedding(question_text)

    # Reference similarity features
    ref_embeddings = _ref_embedding_cache.get(qid)

    if ref_embeddings is not None and len(ref_embeddings) > 0:
        # Max similarity across all reference answers
        sims = [_cosine_similarity(student_emb, ref_emb) for ref_emb in ref_embeddings]
        max_sim = max(sims)
        mean_sim = np.mean(sims)
    else:
        # Fallback: compare with question text (lower confidence)
        max_sim = _cosine_similarity(student_emb, question_emb) * 0.5
        mean_sim = max_sim

    # Question similarity (always useful)
    q_sim = _cosine_similarity(student_emb, question_emb)

    # Length features
    student_words = student_text.split()
    student_len = np.log1p(len(student_words))

    ref_texts = record.get('reference_answers', [])
    if ref_texts:
        avg_ref_len = np.mean([len(preprocess(r).split()) for r in ref_texts])
        ref_len = np.log1p(avg_ref_len)
        len_ratio = len(student_words) / max(avg_ref_len, 1)
    else:
        ref_len = 0.0
        len_ratio = 1.0

    features = [
        max_sim,
        mean_sim,
        q_sim,
        student_len,
        ref_len,
        min(len_ratio, 3.0),  # cap to avoid outliers
    ]

    # NLI features (if available)
    if use_nli and _nli_model is not None and ref_texts:
        # Use the reference that gave max cosine similarity
        best_ref_idx = int(np.argmax([
            _cosine_similarity(student_emb, ref_emb)
            for ref_emb in ref_embeddings
        ])) if ref_embeddings is not None else 0
        best_ref_text = preprocess(ref_texts[best_ref_idx])

        try:
            nli_scores = _nli_model.predict([(best_ref_text, student_text)])
            if isinstance(nli_scores, np.ndarray) and nli_scores.ndim == 2:
                nli_scores = nli_scores[0]
            features.extend(nli_scores.tolist())
        except Exception:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def extract_features_batch(data: list, use_nli: bool = False) -> np.ndarray:
    """Extract features for a batch of records.

    Returns array of shape (N, n_features).
    """
    # Batch-encode all student answers at once for efficiency
    student_texts = [preprocess(d['student_answer']) for d in data]
    student_embeddings = _get_embeddings_batch(student_texts)

    # Also batch-encode question texts
    question_texts = [preprocess(d['question_text']) for d in data]
    question_embeddings = _get_embeddings_batch(question_texts)

    features_list = []
    for i, record in enumerate(data):
        student_emb = student_embeddings[i]
        question_emb = question_embeddings[i]
        qid = record['question_id']

        ref_embeddings = _ref_embedding_cache.get(qid)

        if ref_embeddings is not None and len(ref_embeddings) > 0:
            sims = [_cosine_similarity(student_emb, ref_emb) for ref_emb in ref_embeddings]
            max_sim = max(sims)
            mean_sim = float(np.mean(sims))
        else:
            max_sim = _cosine_similarity(student_emb, question_emb) * 0.5
            mean_sim = max_sim

        q_sim = _cosine_similarity(student_emb, question_emb)

        student_words = student_texts[i].split()
        student_len = np.log1p(len(student_words))

        ref_texts = record.get('reference_answers', [])
        if ref_texts:
            avg_ref_len = np.mean([len(preprocess(r).split()) for r in ref_texts])
            ref_len = np.log1p(avg_ref_len)
            len_ratio = len(student_words) / max(avg_ref_len, 1)
        else:
            ref_len = 0.0
            len_ratio = 1.0

        feats = [max_sim, mean_sim, q_sim, student_len, ref_len, min(len_ratio, 3.0)]

        # NLI features
        if use_nli and _nli_model is not None and ref_texts:
            best_ref_idx = int(np.argmax([
                _cosine_similarity(student_emb, ref_emb)
                for ref_emb in ref_embeddings
            ])) if ref_embeddings is not None else 0
            best_ref_text = preprocess(ref_texts[best_ref_idx])

            try:
                nli_scores = _nli_model.predict([(best_ref_text, student_texts[i])])
                if isinstance(nli_scores, np.ndarray) and nli_scores.ndim == 2:
                    nli_scores = nli_scores[0]
                feats.extend(nli_scores.tolist())
            except Exception:
                feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend([0.0, 0.0, 0.0])

        features_list.append(feats)

    return np.array(features_list, dtype=np.float32)


if __name__ == '__main__':
    print("Features module loaded. Use init_models() + extract_features_batch() in pipeline.")
