"""
chunker.py

Split annual reports into overlapping chunks for RAG without importing
optional machine-learning packages during application startup.
"""

from __future__ import annotations

from langchain_core.documents import Document

DEFAULT_SEPARATORS = ("\n\n", "\n", ".", " ")


def _find_boundary(
    text: str,
    start: int,
    end: int,
    separators: tuple[str, ...],
) -> int:
    """Find a nearby natural boundary without exceeding the chunk limit."""
    for separator in separators:
        boundary = text.rfind(separator, start, end)
        if boundary > start:
            return boundary + len(separator)
    return end


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: tuple[str, ...],
) -> list[str]:
    """Split text into bounded overlapping chunks using natural boundaries."""
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            end = _find_boundary(text, start, end, separators)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split LangChain documents into bounded chunks with copied metadata.

    Args:
        documents: Source pages to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Maximum overlap in characters between chunks.

    Returns:
        Chunked documents retaining each source document's metadata.

    Raises:
        ValueError: If chunk size or overlap values are invalid.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size.")

    chunks: list[Document] = []
    for document in documents:
        chunks.extend(
            Document(page_content=chunk, metadata=document.metadata.copy())
            for chunk in _split_text(
                document.page_content,
                chunk_size,
                chunk_overlap,
                DEFAULT_SEPARATORS,
            )
        )
    return chunks
