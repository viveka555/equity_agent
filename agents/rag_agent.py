"""
rag_agent.py

RAG Agent for the Institutional Equity Research System.

Responsibilities
----------------
- Receive the current LangGraph state.
- Extract the user's question.
- Execute the annual-report RAG chain.
- Store the generated answer in graph state.

Used by
--------
graph.py
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from rag.rag_chain import answer_question
from state import GraphState

logger = logging.getLogger(__name__)

def rag_node(state:GraphState) -> GraphState:
    """
    Execute the annual-report RAG workflow.

    Args:
        state: Current LangGraph state.

    Returns:
        Updated graph state containing the RAG answer.

    Raises:
        ValueError: If no human question is available.
    """
    logger.info(
        "RAG Agent started for company: %s",
        state.get("company", "unknown"),
    )

    last_message = state["messages"][-1]

    if not isinstance(last_message, HumanMessage):
        raise ValueError("RAG Agent requires the last message to be a HumanMessage.")

    question = last_message.content

    answer = answer_question(question)

    return {
        **state,
        "annual_report_analysis": {
            "question": question,
            "answer": answer,
        },
        "final_report": answer,
    }
