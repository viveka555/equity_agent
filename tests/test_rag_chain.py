"""
test_rag_chain.py

Tests for the annual-report RAG chain.

Responsibilities
----------------
- Verify retrieved documents are converted into usable context.
- Verify the complete RAG pipeline can answer a question.
"""

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

    This test uses the actual ChromaDB vector store and Groq LLM.

    Raises:
        AssertionError: If the RAG pipeline does not return
            a non-empty answer.
    """
    question = "What are the major risks faced by Bharat Electronics?"

    answer = answer_question(
        question=question,
        k=5,
    )

    assert answer.strip(), "RAG chain returned an empty answer."