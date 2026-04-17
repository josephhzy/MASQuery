"""
RAGAS-inspired evaluation metrics for RAG quality assessment.

Each metric returns a float in [0, 1]:
- context_relevance:  Do the retrieved chunks actually relate to the question?
- answer_faithfulness: Is the answer grounded in the provided context (no hallucination)?
- answer_correctness:  Does the answer match the expected golden answer?

All metrics use lightweight heuristics (embedding similarity, token overlap) so they
can run without an LLM call. For production evaluation, consider supplementing with
LLM-as-judge approaches.
"""

import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer, util

from config import EMBEDDING_MODEL

# Reuse the project's embedding model for consistency
_eval_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (shared singleton)."""
    global _eval_model
    if _eval_model is None:
        _eval_model = SentenceTransformer(EMBEDDING_MODEL)
    return _eval_model


def _tokenize(text: str) -> set[str]:
    """Simple word-level tokenizer for overlap metrics."""
    return set(re.findall(r"\w+", text.lower()))


# ---------------------------------------------------------------------------
# Metric 1: Context Relevance
# ---------------------------------------------------------------------------


def context_relevance(question: str, contexts: List[str]) -> float:
    """
    Measure how relevant the retrieved contexts are to the question.

    Computes cosine similarity between the question embedding and each
    context embedding, then returns the mean similarity.

    Args:
        question: The user's question.
        contexts: List of retrieved chunk texts.

    Returns:
        Score in [0, 1]. Higher = more relevant contexts.
    """
    if not contexts:
        return 0.0

    model = _get_model()
    q_emb = model.encode(question, normalize_embeddings=True)
    c_embs = model.encode(contexts, normalize_embeddings=True)

    similarities = util.cos_sim(q_emb, c_embs).numpy().flatten()
    # Clamp negatives to 0 (irrelevant contexts shouldn't reduce the score)
    similarities = np.clip(similarities, 0.0, 1.0)
    return float(np.mean(similarities))


# ---------------------------------------------------------------------------
# Metric 2: Answer Faithfulness
# ---------------------------------------------------------------------------


def answer_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Measure whether the answer is grounded in the provided contexts.

    Two-pronged approach:
    1. Sentence-level: embed each answer sentence, find max similarity to any context.
    2. Token-level: what fraction of answer tokens appear in the contexts.

    Returns the average of both signals.

    Args:
        answer: The generated answer text.
        contexts: List of retrieved chunk texts.

    Returns:
        Score in [0, 1]. Higher = more faithful (less hallucination risk).
    """
    if not answer.strip() or not contexts:
        return 0.0

    # Split answer into sentences
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    sentences = [s for s in sentences if len(s.split()) >= 3]  # skip tiny fragments
    if not sentences:
        return 0.0

    model = _get_model()
    context_joined = " ".join(contexts)

    # --- Embedding-based faithfulness ---
    s_embs = model.encode(sentences, normalize_embeddings=True)
    c_embs = model.encode(contexts, normalize_embeddings=True)

    # For each answer sentence, take max similarity to any context chunk
    sim_matrix = util.cos_sim(s_embs, c_embs).numpy()  # (n_sentences, n_contexts)
    max_sims = np.clip(sim_matrix.max(axis=1), 0.0, 1.0)
    embedding_score = float(np.mean(max_sims))

    # --- Token overlap faithfulness ---
    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(context_joined)

    if not answer_tokens:
        return embedding_score

    overlap = answer_tokens & context_tokens
    token_score = len(overlap) / len(answer_tokens)

    return (embedding_score + token_score) / 2.0


# ---------------------------------------------------------------------------
# Metric 3: Answer Correctness
# ---------------------------------------------------------------------------


def answer_correctness(answer: str, expected_answer: str) -> float:
    """
    Measure how well the generated answer matches the expected (golden) answer.

    Combines:
    1. Semantic similarity (embedding cosine similarity)
    2. Token-level F1 score (precision + recall of words)

    Args:
        answer: The generated answer text.
        expected_answer: The golden/expected answer text.

    Returns:
        Score in [0, 1]. Higher = closer to expected answer.
    """
    if not answer.strip() or not expected_answer.strip():
        return 0.0

    model = _get_model()

    # --- Semantic similarity ---
    a_emb = model.encode(answer, normalize_embeddings=True)
    e_emb = model.encode(expected_answer, normalize_embeddings=True)
    semantic_sim = float(np.clip(util.cos_sim(a_emb, e_emb).item(), 0.0, 1.0))

    # --- Token F1 ---
    a_tokens = _tokenize(answer)
    e_tokens = _tokenize(expected_answer)

    if not a_tokens or not e_tokens:
        return semantic_sim

    common = a_tokens & e_tokens
    precision = len(common) / len(a_tokens) if a_tokens else 0.0
    recall = len(common) / len(e_tokens) if e_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Weight semantic similarity more heavily (it captures meaning, not just words)
    return 0.6 * semantic_sim + 0.4 * f1
