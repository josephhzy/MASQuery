"""
Source tracing: extract and verify citations from Claude's response.
Maps citations back to provided chunks to detect hallucinated references.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from retriever import RetrievalResult

logger = logging.getLogger(__name__)

# Regex to extract citations — permissive to handle LLM format variations.
#
# Accepted forms (case-insensitive, whitespace-tolerant around punctuation):
#   [Source: <doc>, Section: <sec>, Page: <pg>]       full form, square brackets
#   (Source: <doc>, Section: <sec>, Page: <pg>)       same, parentheses
#   [Source: <doc>, Section: <sec>]                   page omitted
#   [  Source :  <doc> , Section: <sec> ]             whitespace around colons/commas
#   [Source: <doc>, Sec.: <sec>, Pg. <pg>]            Section/Sec./Part + Page/Pages/Pg./p. abbreviations
#   Page delimiter tolerates colon-with-space OR bare-space after the page label.
#
# The overall shape is a single bracketed (or paren-wrapped) citation that starts
# with "Source:". Section is required (it's the primary verifier signal); page is
# optional. If page is missing, _find_matching_result falls back to doc + section
# matching (page overlap contributes +1 when present; see CITATION_MATCH_THRESHOLD).
#
# See docs/VERIFICATION.md for accepted formats and failure modes.
CITATION_PATTERN = re.compile(
    r'[\[\(]\s*Source\s*:\s*(?P<document>[^,\]\)]+?)\s*'
    r',\s*(?:Section|Sec\.?|Part)\s*:\s*(?P<section>[^,\]\)]+?)\s*'
    r'(?:,\s*(?:Pages?|Pg\.?|p\.?)\s*(?::\s*|\s+)(?P<page>[^\]\)]+?))?\s*[\]\)]',
    re.IGNORECASE
)


# Minimum match score for a citation to be considered "verified".
# Scoring scheme (see _find_matching_result):
#   +2 for doc-name substring containment (either direction) — required, otherwise skip
#   +1 for doc-name word overlap > 0.5 (fallback if no containment)
#   +2 for section-header substring containment (either direction)
#   +1 for section-header word overlap > 0.3
#   +1 for non-empty intersection between citation pages and chunk pages
# Max achievable total is 5 (2 doc + 2 section + 1 page).
# Threshold = 3 means: doc-match (2) AND at least a weak section or page signal (1),
# OR doc-match (2) AND section-match (+2) even if pages are absent or disjoint.
# Weaker signals (doc-match only, or doc-word-overlap + anything) fall below and
# are returned as unverified — the safer default for a compliance tool.
CITATION_MATCH_THRESHOLD = 3


@dataclass
class SourceReference:
    """A verified or unverified source citation from Claude's response."""
    document: str
    section: str
    page_numbers: List[int]
    chunk_id: str
    relevance_score: float
    verified: bool
    text_excerpt: str


def extract_citations(response_text: str) -> List[dict]:
    """
    Extract citation patterns from Claude's response text.

    Returns:
        List of dicts with document, section, page keys.
    """
    citations = []
    seen = set()

    for match in CITATION_PATTERN.finditer(response_text):
        doc = match.group("document").strip()
        section = match.group("section").strip()
        page_group = match.group("page")
        # The page clause is optional in the regex (see CITATION_PATTERN);
        # a missing page group is a legitimate malformed-but-intended cite,
        # not a parse failure.
        page_str = page_group.strip() if page_group else ""

        pages = _parse_pages(page_str) if page_str else []

        key = (doc.lower(), section.lower(), tuple(pages))
        if key in seen:
            continue
        seen.add(key)

        citations.append({
            "document": doc,
            "section": section,
            "pages": pages,
        })

    return citations


def _parse_pages(page_str: str) -> List[int]:
    """Parse page string into list of integers."""
    pages = []
    parts = re.split(r'[,;]', page_str)
    for part in parts:
        part = part.strip()
        range_match = re.match(r'(\d+)\s*-\s*(\d+)', part)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            pages.extend(range(start, end + 1))
        elif part.isdigit():
            pages.append(int(part))
    return sorted(set(pages)) if pages else []


def verify_citations(
    citations: List[dict],
    provided_results: List[RetrievalResult],
) -> List[SourceReference]:
    """
    Cross-reference citations against the chunks actually provided to Claude.
    Unverified citations = hallucination signal.
    """
    references = []

    for citation in citations:
        match = _find_matching_result(citation, provided_results)

        if match:
            references.append(SourceReference(
                document=citation["document"],
                section=citation["section"],
                page_numbers=citation["pages"],
                chunk_id=match.chunk_id,
                relevance_score=match.relevance_score,
                verified=True,
                text_excerpt=match.text[:200] + "..." if len(match.text) > 200 else match.text,
            ))
        else:
            logger.warning(
                f"Unverified citation: {citation['document']}, "
                f"Section: {citation['section']}, Page: {citation['pages']}"
            )
            references.append(SourceReference(
                document=citation["document"],
                section=citation["section"],
                page_numbers=citation["pages"],
                chunk_id="unverified",
                relevance_score=0.0,
                verified=False,
                text_excerpt="",
            ))

    return references


def _find_matching_result(
    citation: dict,
    results: List[RetrievalResult],
) -> Optional[RetrievalResult]:
    """
    Find the RetrievalResult that best matches a citation.
    Uses fuzzy matching on document name and section header.
    """
    citation_doc = citation["document"].lower().strip()
    citation_section = citation["section"].lower().strip()

    best_match = None
    best_score = 0

    for result in results:
        doc_score = 0
        other_score = 0
        result_doc = result.doc_name.lower().strip()
        result_section = result.section_header.lower().strip()

        if citation_doc in result_doc or result_doc in citation_doc:
            doc_score = 2
        elif _word_overlap(citation_doc, result_doc) > 0.5:
            doc_score = 1

        if doc_score == 0:
            continue

        if citation_section in result_section or result_section in citation_section:
            other_score += 2
        elif _word_overlap(citation_section, result_section) > 0.3:
            other_score += 1

        citation_pages = set(citation.get("pages", []))
        result_pages = set(result.page_numbers)
        if citation_pages & result_pages:
            other_score += 1

        total = doc_score + other_score
        if total > best_score:
            best_score = total
            best_match = result

    return best_match if best_score >= CITATION_MATCH_THRESHOLD else None


def _word_overlap(a: str, b: str) -> float:
    """Compute word overlap ratio between two strings."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    return len(intersection) / min(len(words_a), len(words_b))


def trace_response(
    response_text: str,
    provided_results: List[RetrievalResult],
) -> List[SourceReference]:
    """
    Full tracing pipeline: extract citations and verify against provided chunks.
    """
    citations = extract_citations(response_text)

    if not citations:
        logger.info(
            "No inline citations found in response. "
            "Returning retrieved chunks as unverified sources."
        )
        return [
            SourceReference(
                document=r.doc_name,
                section=r.section_header,
                page_numbers=r.page_numbers,
                chunk_id=r.chunk_id,
                relevance_score=r.relevance_score,
                verified=False,
                text_excerpt=r.text[:200] + "..." if len(r.text) > 200 else r.text,
            )
            for r in provided_results
        ]

    return verify_citations(citations, provided_results)
