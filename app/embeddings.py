"""Local, API-key-free embeddings via sentence-transformers.

Kept independent of the chat LLM on purpose: embedding the catalog must not
depend on an LLM provider quota, and it means the vector store can be
rebuilt offline with zero API cost.
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Returns an (n, dim) float32 array, L2-normalized (so dot product == cosine similarity)."""
    if not texts:
        return np.zeros((0, _model().get_sentence_embedding_dimension()), dtype="float32")
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
