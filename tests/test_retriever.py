"""Unit tests for retriever configuration and document retrieval."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag.retriever import get_retriever, retrieve_documents


def test_get_retriever_uses_similarity_search() -> None:
    """Verify the configured top-k is passed to the local vector store."""
    fake_retriever = MagicMock()
    fake_store = MagicMock()
    fake_store.as_retriever.return_value = fake_retriever

    with patch("rag.retriever.load_vector_store", return_value=fake_store):
        retriever = get_retriever(k=3)

    assert retriever is fake_retriever
    fake_store.as_retriever.assert_called_once_with(
        search_type="similarity",
        search_kwargs={"k": 3},
    )


def test_retrieve_documents_uses_query_and_returns_documents() -> None:
    """Verify the caller's query reaches the retriever."""
    query = "What risks does the annual report describe?"
    expected_documents = [
        Document(
            page_content="Risk disclosure text.",
            metadata={"source": "annual-report.pdf", "page": 4},
        )
    ]
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = expected_documents

    with patch("rag.retriever.get_retriever", return_value=fake_retriever):
        documents = retrieve_documents(query, k=2)

    fake_retriever.invoke.assert_called_once_with(query)
    assert documents == expected_documents


@pytest.mark.parametrize("query", ["", "  "])
def test_retrieve_documents_rejects_empty_query(query: str) -> None:
    """Verify empty retrieval queries fail before invoking a retriever."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieve_documents(query)
