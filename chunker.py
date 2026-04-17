"""
Section-aware chunking for MAS regulatory documents.
Preserves section structure and attaches metadata to every chunk.
"""

import logging
import re
from dataclasses import dataclass
from typing import List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_MIN_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
)
from ingest import PageData

logger = logging.getLogger(__name__)

# Token encoder — cl100k_base is close enough for sizing purposes
_encoder = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """A text chunk with full provenance metadata."""
    chunk_id: str
    text: str
    doc_name: str
    page_numbers: List[int]
    section_header: str
    chunk_index: int        # position within document
    token_count: int


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_encoder.encode(text))


def _sanitize_for_id(text: str) -> str:
    """Create a filesystem-safe string for chunk IDs."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', text)[:50]


def _build_sections(pages: List[PageData]) -> List[dict]:
    """
    Group pages into sections based on detected headers.

    Each section records:
    - header: the section title
    - text: all page text (+ tables) joined together
    - pages: ordered list of page numbers in this section
    - page_offsets: list of (char_offset, page_number) so downstream code
      can map a character position back to a specific page number.

    Tables are appended after body text on the same page so they are
    included in retrieval instead of being silently discarded.
    """
    sections = []
    current_header = "Introduction"
    current_text_parts = []
    current_pages = []
    current_page_offsets: List[tuple] = []  # (char_start_of_this_page, page_number)

    for page in pages:
        # Merge body text and any extracted tables for this page.
        # Tables from ingest.PageData.tables are structured pipe-delimited
        # text (see ingest._extract_tables). Wrap each one in explicit
        # [Table start]/[Table end] markers so the LLM can recognise the
        # boundary and the retriever can match table-specific queries.
        page_content = page.text
        if page.tables:
            table_block = "\n\n".join(
                f"[Table start]\n{t}\n[Table end]" for t in page.tables
            )
            page_content = f"{page_content}\n\n{table_block}"

        if page.section_headers:
            # New section header — flush the current section first.
            if current_text_parts:
                sections.append({
                    "header": current_header,
                    "text": "\n\n".join(current_text_parts),
                    "pages": list(current_pages),
                    "page_offsets": list(current_page_offsets),
                })
            current_header = page.section_headers[-1]  # use most specific header
            current_text_parts = [page_content]
            current_pages = [page.page_number]
            current_page_offsets = [(0, page.page_number)]
        else:
            # Continue current section — track char offset where this page starts.
            char_offset = sum(len(p) + 2 for p in current_text_parts)  # +2 for "\n\n"
            current_text_parts.append(page_content)
            if page.page_number not in current_pages:
                current_pages.append(page.page_number)
                current_page_offsets.append((char_offset, page.page_number))

    # Flush last section
    if current_text_parts:
        sections.append({
            "header": current_header,
            "text": "\n\n".join(current_text_parts),
            "pages": list(current_pages),
            "page_offsets": list(current_page_offsets),
        })

    return sections


def _get_pages_for_chunk(
    chunk_text: str,
    section_text: str,
    page_offsets: List[tuple],
) -> List[int]:
    """
    Determine which pages a chunk spans using character-offset mapping.

    Locates the chunk inside the section text, then intersects the chunk's
    character range with each page's character range.  Falls back to all
    section pages if the chunk cannot be located (e.g. after heavy whitespace
    stripping by the splitter).

    Args:
        chunk_text: The raw split text (without the section-header prefix).
        section_text: The full section text the chunk was split from.
        page_offsets: List of (char_start, page_number) pairs in order.

    Returns:
        Sorted list of page numbers this chunk overlaps with.
    """
    if not page_offsets:
        return []
    if len(page_offsets) == 1:
        return [page_offsets[0][1]]

    # Use the first 80 chars as an anchor to find the chunk in the section.
    anchor = chunk_text[:80].strip()
    start = section_text.find(anchor)
    if start == -1:
        # Anchor not found — this happens when the splitter strips whitespace
        # differently from the section text. Fall back to all section pages.
        # This over-reports page span but never drops a page.
        logger.warning(
            "Page anchor not found in section text; falling back to full "
            "section page range. Chunk starts with: %r", anchor[:40]
        )
        return [po[1] for po in page_offsets]

    end = start + len(chunk_text)

    pages = set()
    for i, (offset, page_num) in enumerate(page_offsets):
        next_offset = (
            page_offsets[i + 1][0] if i + 1 < len(page_offsets) else len(section_text)
        )
        # Does this page's character range overlap with [start, end)?
        if offset < end and next_offset > start:
            pages.add(page_num)

    return sorted(pages) if pages else [page_offsets[0][1]]


def chunk_document(pages: List[PageData]) -> List[Chunk]:
    """
    Chunk a document's pages into semantically meaningful pieces.

    Each chunk:
    - Respects section boundaries
    - Gets its section header prepended for self-contained retrieval
    - Carries full metadata (doc, pages, section, ID)

    Args:
        pages: List of PageData from a single document.

    Returns:
        List of Chunk objects with metadata.
    """
    if not pages:
        return []

    doc_name = pages[0].doc_name
    sections = _build_sections(pages)

    # Configure splitter with token-based sizing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=count_tokens,
        strip_whitespace=True,
    )

    chunks = []
    chunk_idx = 0

    for section in sections:
        header = section["header"]
        section_text = section["text"]
        page_offsets = section.get("page_offsets", [])
        fallback_pages = section["pages"]

        # Split section text
        splits = splitter.split_text(section_text)

        for split_text in splits:
            # Prepend section header for self-contained retrieval
            prefixed_text = f"[Section: {header}]\n\n{split_text}"
            token_count = count_tokens(prefixed_text)

            # Skip tiny chunks (likely artifacts)
            if token_count < CHUNK_MIN_SIZE:
                continue

            # Precise page span: only the pages this specific chunk covers,
            # not the entire section's page range.
            chunk_pages = _get_pages_for_chunk(split_text, section_text, page_offsets)
            if not chunk_pages:
                chunk_pages = fallback_pages  # safety fallback

            safe_name = _sanitize_for_id(doc_name)
            chunk_id = f"{safe_name}_p{chunk_pages[0]}_c{chunk_idx}"

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=prefixed_text,
                doc_name=doc_name,
                page_numbers=chunk_pages,
                section_header=header,
                chunk_index=chunk_idx,
                token_count=token_count,
            ))
            chunk_idx += 1

    logger.info(f"Chunked {doc_name}: {len(chunks)} chunks from {len(sections)} sections")
    return chunks


def chunk_all_documents(pages_by_doc: dict[str, List[PageData]]) -> List[Chunk]:
    """
    Chunk multiple documents.

    Args:
        pages_by_doc: Dict mapping doc_name -> list of PageData.

    Returns:
        All chunks across all documents.
    """
    all_chunks = []
    for doc_name, pages in pages_by_doc.items():
        doc_chunks = chunk_document(pages)
        all_chunks.extend(doc_chunks)
    logger.info(f"Total chunks across all documents: {len(all_chunks)}")
    return all_chunks
