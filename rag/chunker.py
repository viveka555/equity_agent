"""
chunker.py

Split annual reports into overlapping chunks for RAG.

Responsibilities
----------------
- Split LangChain Documents
- Preserve metadata (page numbers)
- Create chunks suitable for embeddings
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
#Preserves paragraphs before breaking sentences

from langchain_core.documents import Document

def split_documents(documents: list[Document],
                   chunk_size: int=800,
                   chunk_overlap:int=150)-> list[Document]:
    """Split documents into overlapping chunks.

    Args:
        documents: Pages loaded from the PDF.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunked Document objects."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_documents(documents)