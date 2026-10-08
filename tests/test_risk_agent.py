"""
test_risk_agent.py

Tests for evidence-based Risk Analysis Agent orchestration.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agents.risk_agent import risk_node
from models.planner_models import PlannerDecision
from models.risk_models import RiskAnalysis, RiskFinding


def _risk_analysis() -> RiskAnalysis:
    """Build a representative validated risk result for agent tests."""
    return RiskAnalysis(
        overall_risk_level="moderate",
        summary="Debt is present, while the supplied liquidity ratio is above one.",
        key_risks=[
            RiskFinding(
                category="financial",
                severity="moderate",
                title="Debt exposure",
                evidence="The supplied debt-to-equity ratio is 0.6.",
                source="ratio_analysis",
                implications="Higher financing costs could reduce earnings.",
                mitigants=["The supplied current ratio is 1.4."],
            )
        ],
        data_gaps=["No debt maturity schedule was supplied."],
    )


def test_risk_node_uses_supplied_evidence_and_updates_state() -> None:
    """Verify structured risk output is stored and rendered for users."""
    state = {
        "messages": [
            HumanMessage(content="Assess the financial risks for Example Co.")
        ],
        "company": "Example Co",
        "task": "risk",
        "news": "",
        "final_report": "",
        "ratio_analysis": {
            "debt_to_equity": 0.6,
            "current_ratio": 1.4,
        },
    }
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _risk_analysis()

    with patch("agents.risk_agent.llm") as mock_llm:
        mock_llm.with_structured_output.return_value = structured_llm
        updated_state = risk_node(state)  # type: ignore[arg-type]

    mock_llm.with_structured_output.assert_called_once_with(RiskAnalysis)
    structured_llm.invoke.assert_called_once()
    human_prompt = structured_llm.invoke.call_args.args[0][1][1]
    request = json.loads(human_prompt)
    assert request["company"] == "Example Co"
    assert request["evidence"]["ratio_analysis"]["debt_to_equity"] == 0.6

    assert updated_state["risk_analysis"]["overall_risk_level"] == "moderate"
    assert "Risk Analysis — Example Co" in updated_state["final_report"]
    assert "No debt maturity schedule was supplied." in updated_state[
        "final_report"
    ]


def test_risk_node_rejects_missing_evidence() -> None:
    """Verify the LLM is not asked to assess a company without evidence."""
    state = {
        "messages": [HumanMessage(content="Assess company risks")],
        "company": "Example Co",
        "task": "risk",
        "news": "",
        "final_report": "",
    }

    with patch("agents.risk_agent.llm") as mock_llm:
        with pytest.raises(ValueError, match="No risk-analysis evidence"):
            risk_node(state)  # type: ignore[arg-type]

    mock_llm.with_structured_output.assert_not_called()


def test_planner_decision_accepts_risk_task() -> None:
    """Verify risk is a supported planner task value."""
    decision = PlannerDecision(company="Example Co", task="risk")

    assert decision.task == "risk"
