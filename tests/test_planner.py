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
from unittest.mock import MagicMock, patch

from models.planner_models import PlannerDecision
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

    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = PlannerDecision(
        company="BEL",
        task="news",
    )
    with patch("agents.planner.llm") as mock_llm:
        mock_llm.with_structured_output.return_value = mock_structured_llm
        updated_state = planner_node(state)

    mock_llm.with_structured_output.assert_called_once_with(PlannerDecision)
    mock_structured_llm.invoke.assert_called_once()

    assert updated_state["company"].upper() == "BEL"
    assert updated_state["task"] == "news"


def test_planner_routes_share_price_forecast_to_unknown() -> None:
    """Do not misroute a share-price prediction request into DCF."""
    state = {
        "messages": [HumanMessage(content="give me price forcast for SBI")],
        "company": "",
        "task": "",
        "news": "",
        "final_report": "",
    }
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = PlannerDecision(
        company="SBI",
        task="dcf",
    )

    with patch("agents.planner.llm") as mock_llm:
        mock_llm.with_structured_output.return_value = structured_llm
        updated_state = planner_node(state)

    assert updated_state["task"] == "unknown"
