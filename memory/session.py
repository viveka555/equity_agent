"""
session.py

Memory-aware graph invocation for multi-turn local conversations.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.messages import HumanMessage

from memory.conversation_store import ConversationMemory
from state import GraphState


class GraphInvoker(Protocol):
    """Describe the graph invocation interface used by a local session."""

    def invoke(self, state: GraphState) -> GraphState:
        """Run the graph using a shared application state."""
        ...


def invoke_with_memory(
    graph: GraphInvoker,
    user_message: str,
    conversation_id: str,
    memory: ConversationMemory | None = None,
) -> GraphState:
    """Invoke the graph with local history and persist the completed turn.

    Args:
        graph: Compiled LangGraph application.
        user_message: New user message.
        conversation_id: Stable ID used to resume this conversation later.
        memory: Optional SQLite store; the laptop-local default is used
            when omitted.

    Returns:
        Final graph state for the current turn.

    Raises:
        ValueError: If user input, conversation ID, or graph response is
            empty.
    """
    if not user_message.strip():
        raise ValueError("user_message cannot be empty.")

    conversation_memory = memory or ConversationMemory()
    prior_messages = conversation_memory.load_messages(conversation_id)
    state: GraphState = {
        "messages": [*prior_messages, HumanMessage(content=user_message)],
        "company": "",
        "task": "",
        "news": "",
        "final_report": "",
    }

    result = graph.invoke(state)
    final_report = result.get("final_report", "")
    if not final_report.strip():
        raise ValueError("The graph returned an empty final report.")

    conversation_memory.save_turn(
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=final_report,
    )
    return result
