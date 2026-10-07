"""
test_rag_agent.py

Tests for the LangGraph RAG Agent.

Responsibilities
----------------
- Verify the RAG node accepts valid graph state.
- Verify it executes the RAG chain.
- Verify the generated answer is stored in final_report.
"""

from langchain_core.messages import HumanMessage

from agents.rag_agent import rag_node


def test_rag_node() -> None:
    """
    Verify that the RAG agent processes a user question successfully.

    Raises:
        AssertionError: If the RAG agent does not produce an answer.
    """
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "What are the major risks faced by "
                    "Bharat Electronics?"
                )
            )
        ],
        "company": "BEL",
        "task": "rag",
        "news": "",
        "final_report": "",
    }

    updated_state = rag_node(state)

    assert updated_state["final_report"].strip(), (
        "RAG agent did not produce an answer."
    )