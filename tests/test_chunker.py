"""Tests for the chunking module."""

from chunker import Chunk, chunk_document, count_tokens
from config import CHUNK_MIN_SIZE
from ingest import PageData


def _make_pages(doc_name: str = "TestDoc", texts_and_headers=None):
    """Helper to create PageData objects for testing."""
    if texts_and_headers is None:
        texts_and_headers = [
            ("This is the introduction to the regulatory document. " * 20, []),
            ("Section 5.2 covers access control requirements. " * 30, ["5.2 Access Control"]),
            ("Multi-factor authentication should be implemented. " * 25, []),
            ("Section 6.1 covers outsourcing arrangements. " * 30, ["6.1 Outsourcing"]),
        ]

    pages = []
    for i, (text, headers) in enumerate(texts_and_headers):
        pages.append(
            PageData(
                doc_name=doc_name,
                page_number=i + 1,
                text=text,
                section_headers=headers,
            )
        )
    return pages


class TestChunkDocument:
    def test_produces_chunks(self):
        pages = _make_pages()
        chunks = chunk_document(pages)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_metadata_complete(self):
        """Every chunk must have non-empty metadata fields."""
        pages = _make_pages()
        chunks = chunk_document(pages)
        for chunk in chunks:
            assert chunk.chunk_id, "chunk_id must not be empty"
            assert chunk.doc_name == "TestDoc"
            assert len(chunk.page_numbers) > 0
            assert chunk.section_header, "section_header must not be empty"
            assert chunk.text, "text must not be empty"
            assert chunk.token_count > 0

    def test_chunk_preserves_section_header(self):
        """Each chunk should have its section header prepended."""
        pages = _make_pages()
        chunks = chunk_document(pages)
        for chunk in chunks:
            assert "[Section:" in chunk.text, f"Chunk {chunk.chunk_id} missing section header prefix"

    def test_chunk_ids_unique(self):
        pages = _make_pages()
        chunks = chunk_document(pages)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_ordering(self):
        """chunk_index values should be sequential."""
        pages = _make_pages()
        chunks = chunk_document(pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_pages_produce_no_chunks(self):
        chunks = chunk_document([])
        assert chunks == []

    def test_cross_reference_preserved(self):
        """Cross-references like 'As set out in MAS Notice 644' should not be split mid-sentence."""
        text = (
            "As set out in MAS Notice 644, financial institutions must ensure "
            "compliance with the relevant provisions. " * 5
        )
        pages = [
            PageData(
                doc_name="Test",
                page_number=1,
                text=text,
                section_headers=["Cross References"],
            )
        ]
        chunks = chunk_document(pages)
        # At least one chunk should contain the full cross-reference
        found = any("MAS Notice 644" in c.text for c in chunks)
        assert found, "Cross-reference should appear in at least one chunk"

    def test_minimum_chunk_size_enforced(self):
        """Chunks smaller than CHUNK_MIN_SIZE should be discarded."""
        pages = _make_pages()
        chunks = chunk_document(pages)
        for chunk in chunks:
            assert chunk.token_count >= CHUNK_MIN_SIZE, (
                f"Chunk {chunk.chunk_id} has {chunk.token_count} tokens, below minimum {CHUNK_MIN_SIZE}"
            )

    def test_tables_are_consumed_into_chunks(self):
        """
        Regression test for the 'tables extracted but never indexed' bug.

        ingest.py populates PageData.tables via find_tables(); chunker.py
        must fold that structured text into the section content so the
        cleanly-extracted table is what ends up in the retrieval index,
        not the garbled column-interleaved text PyMuPDF produces for the
        same region.
        """
        table_content = "Requirement | Priority | Owner\nMFA | High | IT Security\nPassword Rotation | Medium | HR"
        pages = [
            PageData(
                doc_name="TableDoc",
                page_number=1,
                text="Section 5.2 covers access control requirements. " * 20,
                section_headers=["5.2 Access Control"],
                tables=[table_content],
            )
        ]

        chunks = chunk_document(pages)
        assert chunks, "Expected at least one chunk for a page with a table"

        # At least one chunk must contain the structured table content and
        # the [Table start]/[Table end] markers that delimit it.
        has_table = any(
            "[Table start]" in c.text and "[Table end]" in c.text and "MFA | High" in c.text for c in chunks
        )
        assert has_table, (
            "No chunk contains the extracted table content. "
            "PageData.tables is being dropped on the floor — tables must be "
            "wrapped in [Table start]/[Table end] markers and folded into "
            "the section text so they are retrievable."
        )

    def test_table_only_page_still_produces_chunk(self):
        """A page whose body text is empty but which has a table must still index the table."""
        pages = [
            PageData(
                doc_name="TableOnly",
                page_number=1,
                text="This is a short lead-in to the table. " * 20,
                section_headers=["Appendix A"],
                tables=["Column A | Column B | Column C\nValue 1 | Value 2 | Value 3\nValue 4 | Value 5 | Value 6"],
            )
        ]
        chunks = chunk_document(pages)
        assert any("Value 1 | Value 2 | Value 3" in c.text for c in chunks)


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_known_text(self):
        tokens = count_tokens("Hello world")
        assert tokens > 0
        assert tokens < 10  # "Hello world" is 2-3 tokens
