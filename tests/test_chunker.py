"""Tests for splitting annual-report pages into indexed chunks."""

import pytest
from langchain_core.documents import Document

from rag.chunker import split_documents


def test_split_annual_report_into_chunks() -> None:
    """Verify chunks preserve source metadata and respect size bounds."""
    document = Document(
        page_content=(
            "First paragraph has useful text.\n\n"
            "Second paragraph contains additional research context."
        ),
        metadata={"source": "annual-report.pdf", "page": 1},
    )
    chunks = split_documents(
        [document],
        chunk_size=32,
        chunk_overlap=8,
    )

    assert chunks
    assert chunks[0].page_content.strip()
    assert all(len(chunk.page_content) <= 32 for chunk in chunks)
    assert all(chunk.metadata == document.metadata for chunk in chunks)


def test_split_documents_rejects_invalid_chunk_overlap() -> None:
    """Verify overlap must be smaller than the chunk size."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents(
            [Document(page_content="Example text")],
            chunk_size=8,
            chunk_overlap=8,
        )
