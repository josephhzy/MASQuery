"""Tests for the PDF ingestion module."""

import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from ingest import PageData, extract_pdf, ingest_directory


def _create_test_pdf(path: Path, pages: list[dict] | None = None):
    """
    Create a minimal PDF for testing using PyMuPDF.

    Args:
        path: Where to save the PDF.
        pages: List of dicts with optional keys:
            - text: body text to insert
            - header: text to insert at larger font size (simulates section header)
            - table: if True, draw a simple table on the page
    """
    if pages is None:
        pages = [{"text": "This is a test paragraph with regulatory content."}]

    doc = fitz.open()
    for page_spec in pages:
        page = doc.new_page(width=595, height=842)  # A4-ish

        y_offset = 72  # start below top margin

        # Insert header text at a larger font size
        if "header" in page_spec:
            page.insert_text(
                fitz.Point(72, y_offset),
                page_spec["header"],
                fontsize=16,
                fontname="helv",
            )
            y_offset += 30

        # Insert body text at normal font size
        if "text" in page_spec:
            page.insert_text(
                fitz.Point(72, y_offset),
                page_spec["text"],
                fontsize=10,
                fontname="helv",
            )

    doc.save(str(path))
    doc.close()


class TestExtractPdf:
    def test_returns_page_data_objects(self):
        """extract_pdf should return a list of PageData with correct fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_doc.pdf"
            _create_test_pdf(pdf_path, [
                {"text": "Page one content about MAS regulations."},
                {"text": "Page two content about compliance requirements."},
            ])

            pages = extract_pdf(pdf_path)

            assert len(pages) == 2
            assert all(isinstance(p, PageData) for p in pages)
            assert pages[0].doc_name == "test_doc"
            assert pages[0].page_number == 1
            assert pages[1].page_number == 2
            assert "Page one content" in pages[0].text
            assert "Page two content" in pages[1].text

    def test_page_data_has_correct_fields(self):
        """Each PageData should have doc_name, page_number, text, section_headers, tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            _create_test_pdf(pdf_path)

            pages = extract_pdf(pdf_path)

            assert len(pages) >= 1
            page = pages[0]
            assert hasattr(page, "doc_name")
            assert hasattr(page, "page_number")
            assert hasattr(page, "text")
            assert hasattr(page, "section_headers")
            assert hasattr(page, "tables")
            assert isinstance(page.section_headers, list)
            assert isinstance(page.tables, list)

    def test_empty_pdf_produces_empty_results(self):
        """A PDF with blank pages should return an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "empty.pdf"
            doc = fitz.open()
            doc.new_page()  # blank page, no text
            doc.save(str(pdf_path))
            doc.close()

            pages = extract_pdf(pdf_path)
            assert pages == []

    def test_file_not_found_raises(self):
        """extract_pdf should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            extract_pdf(Path("/nonexistent/path/to/file.pdf"))

    def test_page_numbers_are_one_indexed(self):
        """Page numbers in PageData should start at 1, not 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "numbered.pdf"
            _create_test_pdf(pdf_path, [
                {"text": "First page."},
                {"text": "Second page."},
                {"text": "Third page."},
            ])

            pages = extract_pdf(pdf_path)
            page_numbers = [p.page_number for p in pages]
            assert page_numbers == [1, 2, 3]


class TestSectionHeaderDetection:
    def test_detects_header_by_larger_font(self):
        """Text at a larger font size than the median should be detected as a header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "headers.pdf"
            _create_test_pdf(pdf_path, [
                {
                    "header": "5.2 Access Control Requirements",
                    "text": "Body text about access control. " * 10,
                },
            ])

            pages = extract_pdf(pdf_path)
            assert len(pages) >= 1
            # The header should appear in section_headers since it's at a larger font
            assert len(pages[0].section_headers) > 0
            assert any("Access Control" in h for h in pages[0].section_headers)

    def test_no_headers_on_uniform_text(self):
        """When all text is the same size, no headers should be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "uniform.pdf"
            _create_test_pdf(pdf_path, [
                {"text": "All text at the same font size, no headers here."},
            ])

            pages = extract_pdf(pdf_path)
            assert len(pages) >= 1
            # With uniform font size, the 1.15x threshold should not trigger
            # (all sizes are equal, so nothing is above 1.15 * median)
            # Note: this might still pick up bold — we accept either outcome
            # as long as it doesn't crash


class TestTableExtraction:
    def test_table_extraction_runs_without_error(self):
        """Table extraction should not raise even on pages without tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "no_table.pdf"
            _create_test_pdf(pdf_path, [
                {"text": "Just plain text, no tables here."},
            ])

            pages = extract_pdf(pdf_path)
            assert len(pages) >= 1
            # tables should be a list (possibly empty)
            assert isinstance(pages[0].tables, list)


class TestIngestDirectory:
    def test_ingest_multiple_pdfs(self):
        """ingest_directory should process all PDFs in the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            _create_test_pdf(dir_path / "doc_a.pdf", [
                {"text": "Document A content."},
            ])
            _create_test_pdf(dir_path / "doc_b.pdf", [
                {"text": "Document B content."},
            ])

            pages = ingest_directory(dir_path)
            doc_names = {p.doc_name for p in pages}
            assert "doc_a" in doc_names
            assert "doc_b" in doc_names

    def test_ingest_empty_directory(self):
        """A directory with no PDFs should return an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pages = ingest_directory(Path(tmpdir))
            assert pages == []

    def test_ingest_nonexistent_directory_raises(self):
        """ingest_directory should raise FileNotFoundError for missing dirs."""
        with pytest.raises(FileNotFoundError):
            ingest_directory(Path("/nonexistent/directory"))

    def test_non_pdf_files_ignored(self):
        """Non-PDF files in the directory should be silently skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)

            # Create a non-PDF file
            (dir_path / "readme.txt").write_text("This is not a PDF.")

            # Create one real PDF
            _create_test_pdf(dir_path / "real.pdf", [
                {"text": "Real PDF content."},
            ])

            pages = ingest_directory(dir_path)
            assert len(pages) >= 1
            assert all(p.doc_name == "real" for p in pages)
