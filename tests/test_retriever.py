"""
test_retriever.py

Tests for the annual-report retriever.

Responsibilities
----------------
- Verify the retriever loads successfully.
- Verify relevant annual-report chunks can be retrieved.
- Verify retrieved documents contain content and metadata.
"""

from rag.retriever import get_retriever, retrieve_documents


def test_get_retriever() -> None:
    """
    Verify that the Chroma retriever can be created.

    Raises:
        AssertionError: If the retriever cannot be created.
    """
    retriever = get_retriever()

    assert retriever is not None


def test_retrieve_documents() -> None:
    """
    Verify that relevant annual-report chunks can be retrieved.

    Raises:
        AssertionError: If no relevant documents are returned.
    """
    query = "What are the major risks faced by Bharat Electronics?"

    documents = retrieve_documents(query, k=5)

    assert documents, "Retriever returned no documents."
    assert len(documents) <= 5

    for document in documents:
        assert document.page_content.strip(), (
            "Retrieved document contains empty content."
        )

        assert "source" in document.metadata, (
            "Retrieved document is missing source metadata."
        )

        assert "page" in document.metadata, (
            "Retrieved document is missing page metadata."
        )