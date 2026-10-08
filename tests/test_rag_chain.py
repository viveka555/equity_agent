"""
test_rag_chain.py

Tests for the annual-report RAG chain.

Responsibilities
----------------
- Verify retrieved documents are converted into usable context.
- Verify the complete RAG pipeline can answer a question.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.rag_chain import build_context, answer_question


def test_build_context() -> None:
    """
    Verify that retrieved documents are formatted correctly.

    Raises:
        AssertionError: If source, page, or content is missing
            from the generated context.
    """
    documents = [
        Document(
            page_content="BEL reported strong revenue growth.",
            metadata={
                "source": "BEL_2025.pdf",
                "page": 25,
            },
        )
    ]

    context = build_context(documents)

    assert "BEL_2025.pdf" in context
    assert "Page: 25" in context
    assert "BEL reported strong revenue growth." in context


def test_answer_question() -> None:
    """
    Verify that the complete RAG pipeline generates an answer.

    Retrieval and the LLM are mocked to keep the test local and repeatable.

    Raises:
        AssertionError: If the RAG pipeline does not return
            a non-empty answer.
    """
    question = "What are the major risks faced by Bharat Electronics?"

    document = Document(
        page_content="The annual report describes the major company risks.",
        metadata={"source": "BEL_2025.pdf", "page": 25},
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = "The report lists key risks."
    with (
        patch("rag.rag_chain.retrieve_documents", return_value=[document])
        as mock_retrieve,
        patch("rag.rag_chain.llm", mock_llm),
    ):
        answer = answer_question(question=question, k=5)

    mock_retrieve.assert_called_once_with(query=question, k=5)
    assert answer.strip(), "RAG chain returned an empty answer."
    assert answer == "The report lists key risks."
