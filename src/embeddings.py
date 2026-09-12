import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cpu'):
        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts):
        """
        Encodes a list of texts into embeddings.
        """
        return self.model.encode(texts, show_progress_bar=False)

    def get_dimension(self):
        return self.model.get_sentence_embedding_dimension()
