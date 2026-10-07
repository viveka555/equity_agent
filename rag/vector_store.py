"""
vector_store.py

Persistent Chroma vector database for annual reports.

Responsibilities
----------------
- Create Chroma database
- Store document embeddings
- Prevent duplicate document ingestion
- Load existing database
- Expose retriever for RAG

Used by
--------
retriever.py
rag_agent.py
"""

from __future__ import annotations

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import CHROMA_PATH
from rag.embeddings import get_embedding_model


COLLECTION_NAME = "annual_reports"


def _generate_document_id(document: Document) -> str:
    """
    Generate a deterministic ID for a document chunk.

    The ID is based on the document source, page number,
    and chunk content. Therefore, ingesting the same chunk
    multiple times produces the same ID.

    Args:
        document: Document chunk to identify.

    Returns:
        Stable SHA-256 based document ID.
    """
    source = str(document.metadata.get("source", ""))
    page = str(document.metadata.get("page", ""))
    content = document.page_content

    unique_content = f"{source}|{page}|{content}"

    return hashlib.sha256(
        unique_content.encode("utf-8")
    ).hexdigest()


def build_vector_store(documents: list[Document]) -> Chroma:
    """
    Create or update the persistent Chroma vector database.

    Existing chunks are not duplicated because each chunk receives
    a deterministic document ID.

    Args:
        documents: Chunked annual report documents.

    Returns:
        Persistent Chroma database instance.

    Raises:
        ValueError: If no documents are provided.
    """
    if not documents:
        raise ValueError("Cannot build vector store from empty documents.")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    db = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_embedding_model(),
        collection_name=COLLECTION_NAME,
    )

    document_ids = [
        _generate_document_id(document)
        for document in documents
    ]

    db.add_documents(
        documents=documents,
        ids=document_ids,
    )

    return db


def load_vector_store() -> Chroma:
    """
    Load an existing Chroma vector database.

    Returns:
        Persistent Chroma database instance.
    """
    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=get_embedding_model(),
        collection_name=COLLECTION_NAME,
    )