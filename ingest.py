"""
PDF ingestion pipeline for MAS regulatory documents.
Extracts text with metadata (page numbers, section headers, tables) using PyMuPDF.
"""

import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF

from config import RAW_DIR

logger = logging.getLogger(__name__)


@dataclass
class PageData:
    """Extracted data from a single PDF page with full provenance."""
    doc_name: str
    page_number: int          # 1-indexed
    text: str
    section_headers: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)


def _detect_section_headers(page: fitz.Page) -> List[str]:
    """
    Detect section headers by analyzing font sizes.
    MAS docs use larger/bold fonts for headers vs 9-10pt body text.
    """
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    font_sizes = []
    text_spans = []

    for block in blocks:
        if block.get("type") != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = span.get("size", 0)
                if text and size > 0:
                    font_sizes.append(size)
                    text_spans.append((text, size, span.get("flags", 0)))

    if not font_sizes:
        return []

    median_size = statistics.median(font_sizes)
    headers = []

    for text, size, flags in text_spans:
        # Header heuristic: font size >1.15x median, or bold (flag bit 4)
        is_larger = size > median_size * 1.15
        is_bold = bool(flags & (1 << 4))  # bit 4 = bold

        if (is_larger or (is_bold and size >= median_size)) and len(text) > 2:
            # Clean up: strip numbering artifacts, limit length
            clean = text.strip()
            if len(clean) < 200:  # skip overly long "headers"
                headers.append(clean)

    return headers


def _extract_tables(page: fitz.Page) -> List[str]:
    """
    Extract tables from page as structured text.
    Uses PyMuPDF's built-in table detection with fallback.
    """
    tables = []
    try:
        tab_finder = page.find_tables()
        for table in tab_finder.tables:
            rows = table.extract()
            if not rows:
                continue
            # Format as pipe-delimited text to preserve structure
            lines = []
            for row in rows:
                cells = [str(cell) if cell is not None else "" for cell in row]
                lines.append(" | ".join(cells))
            tables.append("\n".join(lines))
    except Exception as e:
        logger.debug(f"Table extraction failed on page, using fallback: {e}")

    return tables


def extract_pdf(pdf_path: Path) -> List[PageData]:
    """
    Extract all pages from a PDF with metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of PageData objects, one per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_name = pdf_path.stem
    pages = []

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            if not text or not text.strip():
                logger.debug(f"Skipping blank page {page_num + 1} in {doc_name}")
                continue

            headers = _detect_section_headers(page)
            tables = _extract_tables(page)

            pages.append(PageData(
                doc_name=doc_name,
                page_number=page_num + 1,  # 1-indexed
                text=text.strip(),
                section_headers=headers,
                tables=tables,
            ))
    finally:
        doc.close()

    logger.info(f"Extracted {len(pages)} pages from {doc_name}")
    return pages


def ingest_directory(dir_path: Optional[Path] = None) -> List[PageData]:
    """
    Process all PDFs in a directory.

    Args:
        dir_path: Directory containing PDFs. Defaults to data/raw/.

    Returns:
        All PageData across all PDFs.
    """
    dir_path = Path(dir_path) if dir_path else RAW_DIR

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {dir_path}")
        return []

    all_pages = []
    for pdf_path in pdf_files:
        try:
            pages = extract_pdf(pdf_path)
            all_pages.extend(pages)
            logger.info(f"Ingested {pdf_path.name}: {len(pages)} pages")
        except FileNotFoundError:
            logger.error(f"File disappeared before ingestion: {pdf_path.name}")
        except PermissionError:
            logger.error(f"Permission denied reading {pdf_path.name} — check file permissions")
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_path.name} ({type(e).__name__}): {e}")

    logger.info(f"Total: {len(all_pages)} pages from {len(pdf_files)} documents")
    return all_pages
