"""
Embedding wrapper for sentence-transformers.
Singleton model loading, L2-normalized output for cosine similarity via dot product.
"""

import logging
import threading
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """Load embedding model (singleton — thread-safe, cached after first call)."""
    global _model
    # Fast path: already loaded — no lock needed.
    if _model is not None:
        return _model
    # Slow path: first load. Use double-checked locking so only one thread
    # pays the model-download cost even under concurrent startup requests.
    with _model_lock:
        if _model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)
            actual_dim = _model.get_sentence_embedding_dimension()
            if actual_dim != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding dimension mismatch: model {EMBEDDING_MODEL!r} "
                    f"produces {actual_dim}-d vectors but config.EMBEDDING_DIMENSION "
                    f"is {EMBEDDING_DIMENSION}. Update EMBEDDING_DIMENSION in config.py."
                )
            logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a batch of texts. Returns L2-normalized vectors.

    Args:
        texts: List of strings to embed.
        batch_size: Batch size for encoding.

    Returns:
        numpy array of shape (N, EMBEDDING_DIMENSION), float32, L2-normalized.
    """
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,  # L2 normalize so dot product = cosine similarity
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        numpy array of shape (1, EMBEDDING_DIMENSION), float32, L2-normalized.
    """
    model = get_model()
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )
    return np.array(embedding, dtype=np.float32)
