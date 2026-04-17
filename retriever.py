"""
Hybrid retriever: FAISS vector search + BM25 lexical search + Reciprocal Rank Fusion.
Supports three modes: vector-only, bm25-only, or hybrid (default).
Optional cross-encoder reranking for improved precision.
"""

import json
import logging
import re
import shutil
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from chunker import Chunk
from config import (
    BM25_TOP_K_MULTIPLIER,
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    INDEX_VERSION,
    METADATA_PATH,
    RERANK_ENABLED,
    RERANK_MODEL,
    RERANK_TOP_K,
    RRF_K,
    SEARCH_MODE,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from embeddings import embed_query, embed_texts

logger = logging.getLogger(__name__)

# In-memory state
_index: Optional[faiss.Index] = None
_metadata: Optional[List[dict]] = None
_bm25: Optional[BM25Okapi] = None

# Cross-encoder singleton (lazy loaded)
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score and rank."""

    chunk_id: str
    text: str
    doc_name: str
    page_numbers: List[int]
    section_header: str
    relevance_score: float
    rank: int


def _tokenize_for_bm25(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


def _build_bm25_index(metadata: List[dict]) -> BM25Okapi:
    """Build a BM25 index from chunk metadata."""
    corpus = [_tokenize_for_bm25(m["text"]) for m in metadata]
    return BM25Okapi(corpus)


def _get_cross_encoder():
    """Load cross-encoder model (singleton, thread-safe, lazy)."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    with _cross_encoder_lock:
        if _cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading cross-encoder model: {RERANK_MODEL}")
            _cross_encoder = CrossEncoder(RERANK_MODEL)
            logger.info("Cross-encoder model loaded")
    return _cross_encoder


def build_index(chunks: List[Chunk]) -> int:
    """
    Embed all chunks and build a FAISS index + BM25 index. Saves to disk.

    Args:
        chunks: List of Chunk objects to index.

    Returns:
        Number of chunks indexed.
    """
    global _index, _metadata, _bm25

    if not chunks:
        raise ValueError("Cannot build index from empty chunk list")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Embed all chunk texts
    texts = [chunk.text for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    vectors = embed_texts(texts)

    # Build FAISS index (inner product on normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(vectors)

    # Prepare metadata sidecar (same order as FAISS vectors)
    metadata = [asdict(chunk) for chunk in chunks]

    # Backup existing index before overwriting so a failed re-ingest doesn't
    # destroy the last good state.
    if FAISS_INDEX_PATH.exists():
        shutil.copy2(FAISS_INDEX_PATH, FAISS_INDEX_PATH.with_suffix(".index.bak"))
        logger.info("Backed up existing FAISS index to %s", FAISS_INDEX_PATH.with_suffix(".index.bak"))
    if METADATA_PATH.exists():
        shutil.copy2(METADATA_PATH, METADATA_PATH.with_suffix(".json.bak"))
        logger.info("Backed up existing metadata to %s", METADATA_PATH.with_suffix(".json.bak"))

    # Save to disk -- wrap metadata with a version header so load_index can
    # detect stale indexes if chunking or embedding params change in future.
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "w") as f:
        json.dump({"index_version": INDEX_VERSION, "chunks": metadata}, f, indent=2)

    # Update in-memory state
    _index = index
    _metadata = metadata
    _bm25 = _build_bm25_index(metadata)

    logger.info(f"Index built: {index.ntotal} vectors saved to {FAISS_INDEX_PATH}")
    return index.ntotal


def load_index() -> Tuple[faiss.Index, List[dict]]:
    """
    Load FAISS index and metadata from disk into memory.
    Also builds the BM25 index from loaded metadata.

    Returns:
        Tuple of (FAISS index, metadata list).

    Raises:
        FileNotFoundError: If index files don't exist.
    """
    global _index, _metadata, _bm25

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    _index = faiss.read_index(str(FAISS_INDEX_PATH))

    try:
        with open(METADATA_PATH) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Metadata file is corrupted and cannot be parsed: {METADATA_PATH}. "
            f"Re-run POST /ingest to rebuild. Details: {e}"
        ) from e

    # Support both versioned format {"index_version": ..., "chunks": [...]}
    # and the legacy flat-list format for backward compatibility.
    if isinstance(raw, dict):
        stored_version = raw.get("index_version", "unknown")
        if stored_version != INDEX_VERSION:
            logger.warning(
                "Index version mismatch: stored=%s, expected=%s. "
                "Consider re-running POST /ingest to rebuild with current settings.",
                stored_version,
                INDEX_VERSION,
            )
        _metadata = raw["chunks"]
    else:
        logger.warning("Loading legacy (unversioned) metadata. Re-run POST /ingest to upgrade.")
        _metadata = raw

    if len(_metadata) != _index.ntotal:
        raise ValueError(
            f"Metadata/index mismatch: {len(_metadata)} metadata entries "
            f"but {_index.ntotal} FAISS vectors. Re-run POST /ingest."
        )

    # Build BM25 index from loaded metadata
    _bm25 = _build_bm25_index(_metadata)

    logger.info(f"Index loaded: {_index.ntotal} vectors, {len(_metadata)} metadata entries")
    return _index, _metadata


def is_index_loaded() -> bool:
    """Check if the index is currently loaded in memory."""
    return _index is not None and _metadata is not None


def get_index_size() -> int:
    """Return number of vectors in the loaded index, or 0."""
    return _index.ntotal if _index is not None else 0


def _vector_search(query: str, top_k: int) -> List[Tuple[int, float]]:
    """
    FAISS vector similarity search.

    Returns:
        List of (metadata_index, score) tuples, ranked by score descending.
    """
    query_vector = embed_query(query)
    scores, indices = _index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((int(idx), float(score)))
    return results


def _bm25_search(query: str, top_k: int) -> List[Tuple[int, float]]:
    """
    BM25 lexical search over chunk texts.

    Returns:
        List of (metadata_index, score) tuples, ranked by score descending.
    """
    if _bm25 is None:
        return []

    tokens = _tokenize_for_bm25(query)
    scores = _bm25.get_scores(tokens)

    # Get top-k indices by score
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    return results


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[int, float]]],
    k: int = RRF_K,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion: combine multiple ranked lists into one.

    RRF score for document d = sum over all lists of 1 / (k + rank(d))
    where k is a constant (default 60) and rank is 1-indexed position.

    Args:
        ranked_lists: List of ranked result lists, each containing (index, score) tuples.
        k: RRF constant (higher = more weight to lower-ranked results).

    Returns:
        Fused list of (metadata_index, rrf_score) tuples, sorted by RRF score descending.
    """
    rrf_scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, (idx, _score) in enumerate(ranked_list, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def _rerank(query: str, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
    """
    Rerank results using a cross-encoder model for improved relevance.

    Args:
        query: The user's question.
        results: Retrieved results to rerank.
        top_k: Number of results to keep after reranking.

    Returns:
        Reranked list of RetrievalResult, truncated to top_k.
    """
    if not results:
        return results

    cross_encoder = _get_cross_encoder()
    pairs = [[query, r.text] for r in results]
    scores = cross_encoder.predict(pairs)

    # Attach scores and sort
    scored = list(zip(results, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Truncate and reassign ranks and scores
    reranked = []
    for new_rank, (result, ce_score) in enumerate(scored[:top_k], start=1):
        reranked.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                doc_name=result.doc_name,
                page_numbers=result.page_numbers,
                section_header=result.section_header,
                relevance_score=float(ce_score),
                rank=new_rank,
            )
        )

    return reranked


def search(
    query: str,
    top_k: Optional[int] = None,
    mode: Optional[str] = None,
    rerank: Optional[bool] = None,
) -> List[RetrievalResult]:
    """
    Search for chunks relevant to the query using the configured search mode.

    Supports three modes:
    - "vector": FAISS cosine similarity only
    - "bm25": BM25 lexical search only
    - "hybrid": BM25 + FAISS combined via Reciprocal Rank Fusion (default)

    After retrieval, optionally reranks results with a cross-encoder.

    Args:
        query: The user's question.
        top_k: Number of results to return (defaults to config TOP_K).
        mode: Override search mode ("hybrid", "vector", "bm25"). Defaults to config.
        rerank: Override reranking (True/False). Defaults to config RERANK_ENABLED.

    Returns:
        Ranked list of RetrievalResult objects.
    """
    if _index is None or _metadata is None:
        raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

    k = top_k or TOP_K
    k = min(k, _index.ntotal)
    search_mode = mode or SEARCH_MODE
    do_rerank = rerank if rerank is not None else RERANK_ENABLED

    # Retrieve candidates based on search mode
    if search_mode == "vector":
        ranked = _vector_search(query, k)
    elif search_mode == "bm25":
        ranked = _bm25_search(query, k * BM25_TOP_K_MULTIPLIER)
        ranked = ranked[:k]
    else:
        # Hybrid: fetch more candidates from both, then fuse
        vector_results = _vector_search(query, k * 2)
        bm25_results = _bm25_search(query, k * BM25_TOP_K_MULTIPLIER)
        ranked = _reciprocal_rank_fusion([vector_results, bm25_results])

    # Convert to RetrievalResult objects
    results = []
    for rank, (idx, score) in enumerate(ranked[: k * 2 if do_rerank else k], start=1):
        meta = _metadata[idx]

        # IMPORTANT — Layer-1 pre-gate applies ONLY to dense-only, no-rerank mode.
        #
        # SIMILARITY_THRESHOLD (0.3) is a cosine-similarity threshold, so it only
        # makes sense on raw FAISS cosine scores. In every other path the score
        # lives on a different scale:
        #   - BM25: unbounded TF-IDF score, not comparable to cosine.
        #   - Hybrid (RRF): rank-fusion score in (0, ~2/RRF_K], not a similarity.
        #   - Reranked (ANY mode, INCLUDING vector+rerank — the default config):
        #     cross-encoder logits, not cosine. The CE returns signed logits that
        #     are ordinally informative but cannot be compared against 0.3.
        #
        # Therefore the DEFAULT pipeline (vector + rerank) has no Layer-1 pre-gate
        # on relevance score. Refusal on the default path relies on:
        #   1. Layer-1 top-score gate in generator.generate_answer (still runs,
        #      but on the CE-logit scale of the reranked top result — see
        #      docs/REFUSAL_THRESHOLD.md for the asymmetry caveat).
        #   2. Layer-2 post-generation refusal detection in generator._detect_refusal.
        #
        # Do NOT add a bare `score < SIMILARITY_THRESHOLD` check here for the
        # reranked path — it will silently drop real results because CE logits
        # are routinely negative for correct-but-not-perfect matches.
        # See docs/REFUSAL_THRESHOLD.md for the methodology and the upgrade path
        # (per-path threshold calibrated on a labelled set).
        if search_mode == "vector" and not do_rerank:
            if score < SIMILARITY_THRESHOLD:
                continue

        results.append(
            RetrievalResult(
                chunk_id=meta["chunk_id"],
                text=meta["text"],
                doc_name=meta["doc_name"],
                page_numbers=meta["page_numbers"],
                section_header=meta["section_header"],
                relevance_score=float(score),
                rank=rank,
            )
        )

    # Optional cross-encoder reranking
    if do_rerank and results:
        rerank_k = min(top_k or RERANK_TOP_K, len(results))
        results = _rerank(query, results, rerank_k)

    if results:
        logger.info(
            f"Query: '{query[:50]}...' -> {len(results)} results "
            f"(mode={search_mode}, rerank={do_rerank}, top score: {results[0].relevance_score:.3f})"
        )
    else:
        logger.info(f"Query: '{query[:50]}...' -> 0 results (mode={search_mode})")

    return results


def get_indexed_documents() -> List[dict]:
    """
    List all indexed documents with their chunk counts.

    Returns:
        List of dicts with name, doc_type, chunk_count, page_count.
    """
    if _metadata is None:
        return []

    docs = {}
    for meta in _metadata:
        name = meta["doc_name"]
        if name not in docs:
            docs[name] = {
                "name": name,
                "chunk_count": 0,
                "pages": set(),
            }
        docs[name]["chunk_count"] += 1
        docs[name]["pages"].update(meta["page_numbers"])

    return [
        {
            "name": info["name"],
            "chunk_count": info["chunk_count"],
            "page_count": len(info["pages"]),
        }
        for info in docs.values()
    ]
