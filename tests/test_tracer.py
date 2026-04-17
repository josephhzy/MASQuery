"""Tests for the source tracing module."""

from retriever import RetrievalResult
from tracer import (
    CITATION_MATCH_THRESHOLD,
    extract_citations,
    trace_response,
    verify_citations,
    _find_matching_result,
)


def _make_results():
    """Create mock retrieval results for testing."""
    return [
        RetrievalResult(
            chunk_id="trm_p12_c0",
            text="[Section: 5.2 Access Control] Multi-factor authentication requirements...",
            doc_name="TRM Guidelines",
            page_numbers=[12, 13],
            section_header="5.2 Access Control",
            relevance_score=0.85,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="outsourcing_p8_c0",
            text="[Section: 3.1 Notification] Material outsourcing arrangements...",
            doc_name="Outsourcing Guidelines",
            page_numbers=[8],
            section_header="3.1 Notification Requirements",
            relevance_score=0.72,
            rank=2,
        ),
    ]


class TestExtractCitations:
    def test_extract_valid_citation(self):
        text = "Banks must implement MFA [Source: TRM Guidelines, Section: 5.2 Access Control, Page: 12]."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["document"] == "TRM Guidelines"
        assert citations[0]["section"] == "5.2 Access Control"
        assert 12 in citations[0]["pages"]

    def test_extract_multiple_citations(self):
        text = (
            "Per [Source: TRM Guidelines, Section: 5.2, Page: 12], MFA is required. "
            "Additionally, [Source: Outsourcing Guidelines, Section: 3.1, Page: 8] "
            "requires notification."
        )
        citations = extract_citations(text)
        assert len(citations) == 2

    def test_no_citations(self):
        text = "This response has no citations at all."
        citations = extract_citations(text)
        assert citations == []

    def test_duplicate_citations_deduplicated(self):
        text = (
            "First reference [Source: TRM Guidelines, Section: 5.2, Page: 12]. "
            "Same reference again [Source: TRM Guidelines, Section: 5.2, Page: 12]."
        )
        citations = extract_citations(text)
        assert len(citations) == 1

    def test_page_range_parsing(self):
        text = "[Source: TRM Guidelines, Section: 5.2, Page: 12-14]"
        citations = extract_citations(text)
        assert citations[0]["pages"] == [12, 13, 14]

    def test_malformed_citation_ignored(self):
        text = "[Source: incomplete citation without closing bracket"
        citations = extract_citations(text)
        assert citations == []

    def test_citation_without_page(self):
        """Page clause is optional — citation should still extract."""
        text = "See [Source: TRM Guidelines, Section: 5.2 Access Control] for details."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["document"] == "TRM Guidelines"
        assert citations[0]["section"] == "5.2 Access Control"
        assert citations[0]["pages"] == []

    def test_citation_with_parentheses(self):
        """Parenthesised form — some LLMs emit (Source: ...) instead of [Source: ...]."""
        text = "Per (Source: TRM Guidelines, Section: 5.2 Access Control, Page: 12), MFA is required."
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["document"] == "TRM Guidelines"
        assert 12 in citations[0]["pages"]

    def test_citation_with_extra_whitespace(self):
        """Whitespace around colons/commas shouldn't break the parser."""
        text = "[  Source :  TRM Guidelines , Section:  5.2 Access Control ,  Page:  12 ]"
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["document"] == "TRM Guidelines"
        assert citations[0]["section"] == "5.2 Access Control"
        assert 12 in citations[0]["pages"]

    def test_citation_with_abbreviated_labels(self):
        """Accept Sec./Part and Pg./p. abbreviations."""
        text = "[Source: TRM Guidelines, Sec.: 5.2 Access Control, Pg. 12]"
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0]["section"] == "5.2 Access Control"
        assert 12 in citations[0]["pages"]


class TestCitationMatchThreshold:
    """Boundary tests for the CITATION_MATCH_THRESHOLD magic-number extraction."""

    def test_threshold_value_is_three(self):
        """Sanity: threshold is 3 on the 5-point scale (2 doc + 2 section + 1 page)."""
        assert CITATION_MATCH_THRESHOLD == 3

    def test_score_exactly_three_verifies(self):
        """Doc substring (+2) + page overlap (+1) = 3, with section a non-match. Should verify."""
        result = RetrievalResult(
            chunk_id="trm_p12_c0",
            text="body",
            doc_name="TRM Guidelines",
            page_numbers=[12],
            # Section header has zero word overlap with citation section, so
            # section contributes 0 points. Page overlap gives +1.
            section_header="Totally Unrelated Heading XYZ",
            relevance_score=0.85,
            rank=1,
        )
        citation = {
            "document": "TRM Guidelines",  # +2 substring containment
            "section": "qqqq unusedword",  # +0 (no overlap)
            "pages": [12],  # +1 page intersection
        }
        match = _find_matching_result(citation, [result])
        assert match is not None
        assert match.chunk_id == "trm_p12_c0"

    def test_score_exactly_two_does_not_verify(self):
        """Doc substring (+2) only, no section/page signal = 2, below threshold."""
        result = RetrievalResult(
            chunk_id="trm_p12_c0",
            text="body",
            doc_name="TRM Guidelines",
            page_numbers=[99],  # no page intersection
            section_header="Totally Unrelated Heading XYZ",
            relevance_score=0.85,
            rank=1,
        )
        citation = {
            "document": "TRM Guidelines",  # +2
            "section": "qqqq unusedword",  # +0
            "pages": [12],  # +0 (disjoint)
        }
        match = _find_matching_result(citation, [result])
        assert match is None


class TestVerifyCitations:
    def test_verified_citation(self):
        citations = [{"document": "TRM Guidelines", "section": "5.2 Access Control", "pages": [12]}]
        results = _make_results()
        refs = verify_citations(citations, results)
        assert len(refs) == 1
        assert refs[0].verified is True
        assert refs[0].chunk_id == "trm_p12_c0"

    def test_hallucinated_citation(self):
        """Citation for a document NOT in provided chunks should be unverified."""
        citations = [{"document": "MAS Notice 644", "section": "3.1", "pages": [5]}]
        results = _make_results()
        refs = verify_citations(citations, results)
        assert len(refs) == 1
        assert refs[0].verified is False
        assert refs[0].chunk_id == "unverified"

    def test_empty_citations(self):
        refs = verify_citations([], _make_results())
        assert refs == []


class TestTraceResponse:
    def test_full_trace_with_citations(self):
        text = "Banks must use MFA [Source: TRM Guidelines, Section: 5.2 Access Control, Page: 12]."
        results = _make_results()
        refs = trace_response(text, results)
        assert len(refs) >= 1
        assert any(r.verified for r in refs)

    def test_trace_without_citations_returns_chunks_as_unverified(self):
        """When Claude doesn't cite explicitly, chunks are returned as unverified.
        Treating no-citation responses as fully verified would silently weaken
        hallucination detection — the absence of citations is itself a signal."""
        text = "Banks should implement strong authentication measures."
        results = _make_results()
        refs = trace_response(text, results)
        # Should return all provided chunks as references
        assert len(refs) == len(results)
        # All must be unverified — we cannot confirm grounding without citations
        assert all(not r.verified for r in refs)
