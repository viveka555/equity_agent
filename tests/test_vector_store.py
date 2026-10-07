"""
test_vector_store.py

Tests for the ChromaDB vector store.

Responsibilities
----------------
- Load the BEL annual report.
- Split the report into chunks.
- Build the Chroma vector store.
- Verify documents were successfully stored.

Tests
-----
- test_build_vector_store
"""

from rag.chunker import split_documents
from rag.pdf_loader import load_pdf
from rag.vector_store import build_vector_store


PDF_PATH = "data/annual_reports/BEL_2025.pdf"


def test_build_vector_store() -> None:
    """
    Verify that annual-report chunks can be stored in ChromaDB.

    The test loads the BEL annual report, creates chunks, builds
    the persistent vector store, and verifies that Chroma contains
    the expected number of documents.

    Raises:
        AssertionError: If the vector store does not contain
            the expected documents.
    """
    documents = load_pdf(PDF_PATH)

    chunks = split_documents(documents)

    assert chunks, "Document chunking returned no chunks."

    vector_store = build_vector_store(chunks)

    stored_documents = vector_store.get()

    assert stored_documents["ids"], "ChromaDB contains no document IDs."

    assert len(stored_documents["ids"]) == len(chunks), (
        "Number of documents stored in ChromaDB does not match "
        "the number of chunks."
    )