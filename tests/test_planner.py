"""
test_planner.py

Tests for the LLM-powered Planner Agent.

Responsibilities
----------------
- Verify that the planner identifies the company.
- Verify that the planner identifies the requested task.
- Verify that the planner updates LangGraph state correctly.
"""

from langchain_core.messages import HumanMessage

from agents.planner import planner_node


def test_planner_node() -> None:
    """
    Verify that the planner correctly identifies a news request.

    Raises:
        AssertionError: If the planner does not produce the
            expected company or task.
    """
    state = {
        "messages": [
            HumanMessage(content="What is the latest news about BEL?")
        ],
        "company": "",
        "task": "",
        "news": "",
        "final_report": "",
    }

    updated_state = planner_node(state)

    assert updated_state["company"].upper() == "BEL"
    assert updated_state["task"] == "news"