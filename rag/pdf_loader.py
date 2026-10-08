"""
pdf_loader.py

Load annual reports as page-level LangChain documents using pypdf.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """Load PDF pages and preserve source and zero-based page metadata.

    Args:
        pdf_path: Path to the annual report PDF.

    Returns:
        One document per PDF page, including its extracted text and metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        pypdf.errors.PdfReadError: If the file cannot be parsed as a PDF.
    """
    from pypdf import PdfReader

    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages):
        documents.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={
                    "source": str(path),
                    "page": page_number,
                    "total_pages": total_pages,
                },
            )
        )

    return documents
