"""
retriever.py

Retriever configuration for the annual-report RAG pipeline.

Responsibilities
----------------
- Load the persistent Chroma vector store.
- Create a similarity-based retriever.
- Return relevant annual-report chunks for a user question.

Used by
--------
rag_chain.py
agents/rag_agent.py
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from rag.vector_store import load_vector_store


DEFAULT_K = 5


def get_retriever(k: int = DEFAULT_K) -> VectorStoreRetriever:
    """
    Create a retriever from the persistent Chroma vector store.

    The retriever performs similarity search against the embedded
    annual-report chunks and returns the most relevant documents.

    Args:
        k: Number of relevant document chunks to retrieve.

    Returns:
        Configured Chroma vector store retriever.

    Raises:
        ValueError: If k is less than 1.
    """
    if k < 1:
        raise ValueError("k must be greater than or equal to 1.")

    vector_store = load_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def retrieve_documents(
    query: str,
    k: int = DEFAULT_K,
) -> list[Document]:
    """
    Retrieve the most relevant annual-report chunks for a query.

    Args:
        query: User question or search query.
        k: Number of relevant chunks to retrieve.

    Returns:
        List of relevant Document objects.

    Raises:
        ValueError: If the query is empty or k is less than 1.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    retriever = get_retriever(k=k)

    return retriever.invoke(query)