"""Tests for the FastAPI endpoints."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


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
