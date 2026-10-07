"""
test_embeddings.py

Tests for the embedding model used by the RAG pipeline.

Responsibilities
----------------
- Load the BEL annual report.
- Create document chunks.
- Load the embedding model lazily.
- Generate an embedding for a document chunk.
- Verify the embedding vector is valid.
"""

from rag.chunker import split_documents
from rag.embeddings import get_embedding_model
from rag.pdf_loader import load_pdf


PDF_PATH = "data/annual_reports/BEL_2025.pdf"


def test_embedding_generation() -> None:
    """
    Verify that a document chunk can be converted into an embedding.

    Raises:
        AssertionError: If the embedding vector is empty or invalid.
    """
    documents = load_pdf(PDF_PATH)

    chunks = split_documents(documents)

    assert chunks, "Document chunking returned no chunks."

    embedding_model = get_embedding_model()

    vector = embedding_model.embed_query(
        chunks[0].page_content
    )

    assert vector, "Embedding vector is empty."

    assert len(vector) == 384, (
        "Expected a 384-dimensional embedding vector."
    )