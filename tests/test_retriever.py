"""Tests for the retriever module (hybrid search: vector, BM25, RRF, reranking)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from chunker import Chunk


def _make_chunks():
    """Create test chunks about different regulatory topics."""
    return [
        Chunk(
            chunk_id="trm_p1_c0",
            text="[Section: Access Control] Multi-factor authentication should be implemented for all critical systems.",
            doc_name="TRM_Guidelines",
            page_numbers=[12],
            section_header="5.2 Access Control",
            chunk_index=0,
            token_count=20,
        ),
        Chunk(
            chunk_id="trm_p5_c1",
            text="[Section: Data Protection] Financial institutions must encrypt sensitive data at rest and in transit.",
            doc_name="TRM_Guidelines",
            page_numbers=[25],
            section_header="7.1 Data Protection",
            chunk_index=1,
            token_count=18,
        ),
        Chunk(
            chunk_id="outsourcing_p3_c0",
            text="[Section: Outsourcing] Material outsourcing arrangements require prior notification to MAS.",
            doc_name="Outsourcing_Guidelines",
            page_numbers=[8],
            section_header="3.1 Notification Requirements",
            chunk_index=0,
            token_count=16,
        ),
    ]


def _build_test_index(retriever, tmpdir, chunks=None):
    """Helper: build an index in a temp directory with patched paths."""
    if chunks is None:
        chunks = _make_chunks()

    with patch.object(retriever, 'FAISS_INDEX_PATH', Path(tmpdir) / "test.index"), \
         patch.object(retriever, 'METADATA_PATH', Path(tmpdir) / "test_meta.json"), \
         patch.object(retriever, 'INDEX_DIR', Path(tmpdir)):
        retriever._index = None
        retriever._metadata = None
        retriever._bm25 = None
        retriever.build_index(chunks)

    return chunks


class TestBuildAndSearch:
    """Integration tests that build an index and search it."""

    def test_build_and_search(self):
        """Build index from test chunks, search, verify results."""
        import retriever

        chunks = _make_chunks()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(retriever, 'FAISS_INDEX_PATH', Path(tmpdir) / "test.index"), \
                 patch.object(retriever, 'METADATA_PATH', Path(tmpdir) / "test_meta.json"), \
                 patch.object(retriever, 'INDEX_DIR', Path(tmpdir)):

                retriever._index = None
                retriever._metadata = None
                retriever._bm25 = None

                num = retriever.build_index(chunks)
                assert num == 3

                # Search in vector mode (no reranking to keep it simple)
                results = retriever.search(
                    "multi-factor authentication", top_k=3, mode="vector", rerank=False,
                )
                assert len(results) > 0
                assert results[0].chunk_id == "trm_p1_c0", (
                    "Access control chunk should rank highest for MFA query"
                )

                # Verify ranking order
                scores = [r.relevance_score for r in results]
                assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self):
        """Should not return more than top_k results."""
        import retriever

        chunks = _make_chunks()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(retriever, 'FAISS_INDEX_PATH', Path(tmpdir) / "test.index"), \
                 patch.object(retriever, 'METADATA_PATH', Path(tmpdir) / "test_meta.json"), \
                 patch.object(retriever, 'INDEX_DIR', Path(tmpdir)):

                retriever._index = None
                retriever._metadata = None
                retriever._bm25 = None
                retriever.build_index(chunks)

                results = retriever.search("authentication", top_k=1, mode="vector", rerank=False)
                assert len(results) <= 1

    def test_build_index_saves_files(self):
        """Index and metadata files should exist after build."""
        import retriever

        chunks = _make_chunks()

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test.index"
            meta_path = Path(tmpdir) / "test_meta.json"

            with patch.object(retriever, 'FAISS_INDEX_PATH', idx_path), \
                 patch.object(retriever, 'METADATA_PATH', meta_path), \
                 patch.object(retriever, 'INDEX_DIR', Path(tmpdir)):

                retriever._index = None
                retriever._metadata = None
                retriever._bm25 = None
                retriever.build_index(chunks)

                assert idx_path.exists()
                assert meta_path.exists()

                with open(meta_path) as f:
                    meta = json.load(f)
                assert len(meta["chunks"]) == 3
                assert meta["chunks"][0]["chunk_id"] == "trm_p1_c0"

    def test_get_indexed_documents(self):
        """Document listing should match indexed data."""
        import retriever

        chunks = _make_chunks()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(retriever, 'FAISS_INDEX_PATH', Path(tmpdir) / "test.index"), \
                 patch.object(retriever, 'METADATA_PATH', Path(tmpdir) / "test_meta.json"), \
                 patch.object(retriever, 'INDEX_DIR', Path(tmpdir)):

                retriever._index = None
                retriever._metadata = None
                retriever._bm25 = None
                retriever.build_index(chunks)

                docs = retriever.get_indexed_documents()
                names = {d["name"] for d in docs}
                assert "TRM_Guidelines" in names
                assert "Outsourcing_Guidelines" in names


class TestSearchModes:
    """Test the three search modes: vector, bm25, hybrid."""

    def test_bm25_search_returns_results(self):
        """BM25-only search should return relevant results."""
        import retriever

        chunks = _make_chunks()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(retriever, 'FAISS_INDEX_PATH', Path(tmpdir) / "test.index"), \
                 patch.object(retriever, 'METADATA_PATH', Path(tmpdir) / "test_meta.json"), \
                 patch.object(retriever, 'INDEX_DIR', Path(tmpdir)):

                retriever._index = None
                retriever._metadata = None
                retriever._bm25 = None
                retriever.build_index(chunks)

                results = retriever.search(
                    "outsourcing notification MAS", top_k=3, mode="bm25", rerank=False,
                )
                assert len(results) > 0
                # BM25 should find the outsourcing chunk by keyword match
                chunk_ids = [r.chunk_id for r in results]
                assert "outsourcing_p3_c0" in chunk_ids

    def test_hybrid_search_returns_results(self):
        """Hybrid search should combine BM25 and vector results."""
        import retriever

        chunks = _make_chunks()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(retriever, 'FAISS_INDEX_PATH', Path(tmpdir) / "test.index"), \
                 patch.object(retriever, 'METADATA_PATH', Path(tmpdir) / "test_meta.json"), \
                 patch.object(retriever, 'INDEX_DIR', Path(tmpdir)):

                retriever._index = None
                retriever._metadata = None
                retriever._bm25 = None
                retriever.build_index(chunks)

                results = retriever.search(
                    "authentication requirements", top_k=3, mode="hybrid", rerank=False,
                )
                assert len(results) > 0

    def test_bm25_index_built_alongside_faiss(self):
        """BM25 index should be created when building the FAISS index."""
        import retriever

        chunks = _make_chunks()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(retriever, 'FAISS_INDEX_PATH', Path(tmpdir) / "test.index"), \
                 patch.object(retriever, 'METADATA_PATH', Path(tmpdir) / "test_meta.json"), \
                 patch.object(retriever, 'INDEX_DIR', Path(tmpdir)):

                retriever._index = None
                retriever._metadata = None
                retriever._bm25 = None
                retriever.build_index(chunks)

                assert retriever._bm25 is not None


class TestRRF:
    """Test Reciprocal Rank Fusion logic."""

    def test_rrf_combines_two_lists(self):
        """RRF should merge two ranked lists by reciprocal rank scoring."""
        from retriever import _reciprocal_rank_fusion

        list_a = [(0, 0.9), (1, 0.8), (2, 0.7)]
        list_b = [(2, 0.95), (1, 0.6), (3, 0.5)]

        fused = _reciprocal_rank_fusion([list_a, list_b], k=60)

        # All unique indices from both lists should appear
        fused_indices = {idx for idx, _ in fused}
        assert fused_indices == {0, 1, 2, 3}

        # Items appearing in both lists should generally rank higher
        # Index 1 and 2 appear in both lists
        fused_dict = {idx: score for idx, score in fused}
        # Index 2: rank 3 in list_a + rank 1 in list_b -> high combined score
        # Index 0: rank 1 in list_a only
        assert fused_dict[2] > fused_dict[0], "Item in both lists should score higher than one-list item"

    def test_rrf_empty_lists(self):
        """RRF with empty input should return empty."""
        from retriever import _reciprocal_rank_fusion
        assert _reciprocal_rank_fusion([]) == []

    def test_rrf_single_list(self):
        """RRF with a single list should preserve order."""
        from retriever import _reciprocal_rank_fusion

        single = [(5, 0.9), (3, 0.7), (1, 0.5)]
        fused = _reciprocal_rank_fusion([single], k=60)

        fused_indices = [idx for idx, _ in fused]
        assert fused_indices == [5, 3, 1]


class TestEdgeCases:
    def test_empty_index_raises(self):
        import retriever
        with pytest.raises(ValueError):
            retriever.build_index([])

    def test_search_without_load_raises(self):
        import retriever
        retriever._index = None
        retriever._metadata = None
        retriever._bm25 = None
        with pytest.raises(RuntimeError):
            retriever.search("test query")
