"""Tests for the FastAPI endpoints."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from chunker import Chunk
from main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


def _e2e_chunks():
    """Tiny inline corpus for end-to-end pipeline tests."""
    return [
        Chunk(
            chunk_id="trm_p12_c0",
            text=(
                "[Section: 5.2 Access Control] Multi-factor authentication "
                "should be implemented for all critical systems and privileged "
                "accounts. Role-based access control must be enforced and "
                "access rights reviewed regularly."
            ),
            doc_name="TRM_Guidelines",
            page_numbers=[12],
            section_header="5.2 Access Control",
            chunk_index=0,
            token_count=40,
        ),
        Chunk(
            chunk_id="trm_p25_c1",
            text=(
                "[Section: 7.1 Data Protection] Financial institutions must "
                "encrypt sensitive data at rest and in transit, classify data "
                "by sensitivity, and apply data loss prevention controls."
            ),
            doc_name="TRM_Guidelines",
            page_numbers=[25],
            section_header="7.1 Data Protection",
            chunk_index=1,
            token_count=36,
        ),
    ]


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """Health endpoint should always return 200, even when degraded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "faiss_index_loaded" in data
        assert "api_key_configured" in data

    def test_health_reports_no_index(self, client):
        """When no index loaded, health should reflect that."""
        import retriever

        original_index = retriever._index
        retriever._index = None

        response = client.get("/health")
        data = response.json()
        assert data["faiss_index_loaded"] is False

        retriever._index = original_index


class TestDocumentsEndpoint:
    def test_documents_empty_when_no_index(self, client):
        """Should return empty list when no index exists."""
        import retriever

        original_meta = retriever._metadata
        retriever._metadata = None

        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total_documents"] == 0

        retriever._metadata = original_meta


class TestQueryEndpoint:
    def test_query_validation_short_question(self, client):
        """Questions shorter than 10 chars should be rejected."""
        response = client.post("/query", json={"question": "short"})
        assert response.status_code == 422

    def test_query_validation_invalid_search_mode(self, client):
        """Unknown search modes should be rejected instead of silently routing to hybrid."""
        response = client.post(
            "/query",
            json={
                "question": "What are the access control requirements?",
                "search_mode": "semantic-ish",
            },
        )
        assert response.status_code == 422

    def test_query_returns_503_when_no_index(self, client):
        """Should return 503 when FAISS index not loaded."""
        import retriever

        original_index = retriever._index
        original_meta = retriever._metadata
        retriever._index = None
        retriever._metadata = None

        response = client.post("/query", json={"question": "What are the TRM requirements for access control?"})
        assert response.status_code == 503

        retriever._index = original_index
        retriever._metadata = original_meta

    def test_query_response_structure(self, client):
        """Valid query response should match QueryResponse schema."""
        import retriever
        from retriever import RetrievalResult

        mock_results = [
            RetrievalResult(
                chunk_id="test_c0",
                text="[Section: Test] Test content about access control.",
                doc_name="Test_Doc",
                page_numbers=[1],
                section_header="Test Section",
                relevance_score=0.85,
                rank=1,
            )
        ]

        with (
            patch.object(retriever, "is_index_loaded", return_value=True),
            patch.object(retriever, "search", return_value=mock_results),
            patch("main.generate_answer") as mock_gen,
            patch("main.trace_response") as mock_trace,
        ):
            from generator import GenerationResult
            from tracer import SourceReference

            mock_gen.return_value = GenerationResult(
                answer="Test answer about access control.",
                confidence="high",
                is_answerable=True,
                retrieval_scores=[0.85],
            )
            mock_trace.return_value = [
                SourceReference(
                    document="Test_Doc",
                    section="Test Section",
                    page_numbers=[1],
                    chunk_id="test_c0",
                    relevance_score=0.85,
                    verified=True,
                    text_excerpt="Test content...",
                )
            ]

            response = client.post("/query", json={"question": "What are the access control requirements?"})
            assert response.status_code == 200
            data = response.json()

            # Verify response structure
            assert "answer" in data
            assert "confidence" in data
            assert "is_answerable" in data
            assert "sources" in data
            assert "query" in data
            assert "model" in data
            assert "search_mode" in data
            assert "rerank_enabled" in data
            assert data["confidence"] in ("high", "medium", "low")


class TestIngestEndpoint:
    def test_ingest_happy_path(self, client):
        """Happy path: mocked pipeline returns IngestResponse."""
        from unittest.mock import MagicMock

        fake_page = MagicMock()
        fake_page.doc_name = "test_doc"

        with (
            patch("main.ingest_directory", return_value=[fake_page]),
            patch("main.chunk_all_documents", return_value=[]),
            patch("main.retriever.build_index", return_value=2),
        ):
            response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ingested"

    def test_ingest_401_when_token_required_and_missing(self, client, monkeypatch):
        """Should return 401 when INGEST_TOKEN is set but header is absent."""
        import main

        monkeypatch.setattr(main, "INGEST_TOKEN", "secret")
        response = client.post("/ingest")
        assert response.status_code == 401

    def test_ingest_409_when_lock_held(self, client):
        """Should return 409 when _ingest_lock is already acquired."""
        import main

        main._ingest_lock.acquire()
        try:
            response = client.post("/ingest")
            assert response.status_code == 409
        finally:
            main._ingest_lock.release()


class TestEndToEndPipeline:
    """End-to-end /query tests with a real retriever + generator + tracer.

    Only the LLM HTTP call is mocked — `generate_answer` runs in full, the
    real `tracer.trace_response` parses citations from the canned answer, and
    the FastAPI response schema is exercised against actual retrieval data.
    This closes the gap where the rest of the suite mocks the entire
    `main.generate_answer` and `main.trace_response`, leaving the production
    `retrieve -> generate -> trace -> response` chain unverified.
    """

    def test_query_full_pipeline_with_mocked_llm(self, client):
        """Real retriever + real generate_answer + real tracer; LLM mocked.

        The canned LLM response carries a citation in the documented
        ``[Source: <doc>, Section: <sec>, Page: <pg>]`` format so the tracer
        actually verifies it against the retrieved chunks. Asserts the
        response includes a verified source — proof that retrieval found the
        right chunk AND tracer matched the citation to it.
        """
        import generator
        import retriever

        canned_answer = (
            "Financial institutions should implement multi-factor authentication "
            "for privileged accounts and enforce role-based access control "
            "[Source: TRM_Guidelines, Section: 5.2 Access Control, Page: 12]."
        )

        original_index = retriever._index
        original_meta = retriever._metadata
        original_bm25 = retriever._bm25

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with (
                    patch.object(retriever, "FAISS_INDEX_PATH", Path(tmpdir) / "test.index"),
                    patch.object(retriever, "METADATA_PATH", Path(tmpdir) / "test_meta.json"),
                    patch.object(retriever, "INDEX_DIR", Path(tmpdir)),
                ):
                    # Build a real FAISS+BM25 index from our tiny corpus.
                    retriever._index = None
                    retriever._metadata = None
                    retriever._bm25 = None
                    retriever.build_index(_e2e_chunks())

                    # Mock the LLM HTTP call only — generate_answer runs in full.
                    # Patch both providers so the test is provider-agnostic. Clear
                    # the query cache so this test is independent of others.
                    generator._query_cache.clear()

                    with (
                        patch("generator._call_claude", return_value=canned_answer),
                        patch("generator._call_openai", return_value=canned_answer),
                        patch(
                            "generator.load_system_prompt",
                            return_value="System prompt long enough to satisfy minimums.",
                        ),
                    ):
                        response = client.post(
                            "/query",
                            json={
                                "question": "What are the access control requirements under the TRM Guidelines?",
                                "search_mode": "vector",
                                "rerank": False,
                            },
                        )

                    assert response.status_code == 200, response.text
                    data = response.json()

                    # Pipeline-level assertions — each one is checking that a
                    # specific stage actually ran end to end.
                    assert "[Source: TRM_Guidelines" in data["answer"], (
                        "LLM mock answer should be passed through unchanged"
                    )
                    assert data["is_answerable"] is True
                    assert data["search_mode"] == "vector"
                    assert data["rerank_enabled"] is False

                    # Retrieval ran: at least one source returned.
                    assert len(data["sources"]) >= 1, "retriever returned nothing"

                    # Tracer ran: at least one source matched the citation
                    # in the canned answer (verified=True), proving citation
                    # parsing + verification worked end-to-end.
                    verified = [s for s in data["sources"] if s["verified"]]
                    assert verified, "tracer did not verify the inline citation against retrieved chunks"
                    assert verified[0]["document"] == "TRM_Guidelines"
                    assert verified[0]["section"].startswith("5.2 Access Control")
                    assert 12 in verified[0]["page_numbers"]
            finally:
                # Restore module state so other tests aren't polluted by the
                # temp-dir paths or the canned index.
                retriever._index = original_index
                retriever._metadata = original_meta
                retriever._bm25 = original_bm25
                generator._query_cache.clear()
