"""Shared test fixtures for MASQuery tests."""

import tempfile
from pathlib import Path

import pytest

from chunker import Chunk
from retriever import RetrievalResult


@pytest.fixture
def sample_chunks():
    """Create a list of sample Chunk objects for testing."""
    return [
        Chunk(
            chunk_id="trm_p1_c0",
            text="[Section: 5.2 Access Control] Multi-factor authentication should be implemented for all critical systems.",
            doc_name="TRM_Guidelines",
            page_numbers=[12],
            section_header="5.2 Access Control",
            chunk_index=0,
            token_count=20,
        ),
        Chunk(
            chunk_id="trm_p5_c1",
            text="[Section: 7.1 Data Protection] Financial institutions must encrypt sensitive data at rest and in transit.",
            doc_name="TRM_Guidelines",
            page_numbers=[25],
            section_header="7.1 Data Protection",
            chunk_index=1,
            token_count=18,
        ),
        Chunk(
            chunk_id="outsourcing_p3_c0",
            text="[Section: 3.1 Notification] Material outsourcing arrangements require prior notification to MAS.",
            doc_name="Outsourcing_Guidelines",
            page_numbers=[8],
            section_header="3.1 Notification Requirements",
            chunk_index=0,
            token_count=16,
        ),
    ]


@pytest.fixture
def sample_retrieval_result():
    """Create a sample RetrievalResult for testing."""
    return RetrievalResult(
        chunk_id="trm_p12_c0",
        text="[Section: 5.2 Access Control] Multi-factor authentication requirements for critical systems.",
        doc_name="TRM_Guidelines",
        page_numbers=[12, 13],
        section_header="5.2 Access Control",
        relevance_score=0.85,
        rank=1,
    )


@pytest.fixture
def sample_retrieval_results():
    """Create a list of RetrievalResult objects for testing."""
    return [
        RetrievalResult(
            chunk_id="trm_p12_c0",
            text="[Section: 5.2 Access Control] Multi-factor authentication requirements...",
            doc_name="TRM_Guidelines",
            page_numbers=[12, 13],
            section_header="5.2 Access Control",
            relevance_score=0.85,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="outsourcing_p8_c0",
            text="[Section: 3.1 Notification] Material outsourcing arrangements...",
            doc_name="Outsourcing_Guidelines",
            page_numbers=[8],
            section_header="3.1 Notification Requirements",
            relevance_score=0.72,
            rank=2,
        ),
    ]


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def reset_retriever_state():
    """Reset retriever global state before each test to avoid cross-test contamination."""
    import retriever

    original_index = retriever._index
    original_metadata = retriever._metadata
    original_bm25 = retriever._bm25

    yield

    retriever._index = original_index
    retriever._metadata = original_metadata
    retriever._bm25 = original_bm25


@pytest.fixture(autouse=True)
def reset_generator_cache():
    """Clear the generator query cache before each test."""
    import generator

    generator._query_cache.clear()

    yield

    generator._query_cache.clear()
