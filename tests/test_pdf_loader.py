"""Tests for page-level PDF loading and metadata preservation."""

from pathlib import Path

from pypdf import PdfWriter

from rag.pdf_loader import load_pdf


def test_load_pdf_preserves_page_metadata(tmp_path: Path) -> None:
    """Verify the loader reads a small local PDF without external assets."""
    pdf_path = tmp_path / "annual-report.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    documents = load_pdf(pdf_path)

    assert len(documents) == 1
    assert documents[0].metadata == {
        "source": str(pdf_path),
        "page": 0,
        "total_pages": 1,
    }
