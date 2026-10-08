"""
test_conversation_memory.py

Tests for laptop-local conversation history and memory-aware graph calls.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from memory.conversation_store import ConversationMemory
from memory.session import invoke_with_memory


def test_conversation_memory_persists_recent_messages(
    tmp_path: Path,
) -> None:
    """Verify history survives store recreation and is bounded per session."""
    database_path = tmp_path / "memory.sqlite3"
    memory = ConversationMemory(database_path, max_messages=3)
    memory.save_turn("company-research", "First question", "First answer")
    memory.save_turn("company-research", "Follow-up", "Follow-up answer")
    memory.save_turn("another-company", "Other question", "Other answer")

    restored_memory = ConversationMemory(database_path, max_messages=3)
    messages = restored_memory.load_messages("company-research")

    assert [message.content for message in messages] == [
        "Follow-up",
        "Follow-up answer",
    ]
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)


def test_conversation_memory_rejects_blank_identifiers(
    tmp_path: Path,
) -> None:
    """Verify an empty conversation ID cannot mix sessions."""
    memory = ConversationMemory(tmp_path / "memory.sqlite3")

    with pytest.raises(ValueError, match="conversation_id"):
        memory.load_messages("  ")


def test_invoke_with_memory_supplies_history_and_saves_turn(
    tmp_path: Path,
) -> None:
    """Verify graph calls receive prior history and persist their answer."""
    memory = ConversationMemory(tmp_path / "memory.sqlite3")
    memory.save_turn("local", "Analyze BEL", "BEL analysis")
    graph = MagicMock()
    graph.invoke.return_value = {"final_report": "Follow-up answer"}

    result = invoke_with_memory(
        graph=graph,
        user_message="What are the risks?",
        conversation_id="local",
        memory=memory,
    )

    graph_state = graph.invoke.call_args.args[0]
    assert [message.content for message in graph_state["messages"]] == [
        "Analyze BEL",
        "BEL analysis",
        "What are the risks?",
    ]
    assert result["final_report"] == "Follow-up answer"
    assert [message.content for message in memory.load_messages("local")][-2:] == [
        "What are the risks?",
        "Follow-up answer",
    ]
