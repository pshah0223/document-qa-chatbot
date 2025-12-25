import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class Embedder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME, device=None):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=batch_size
            )
        )
