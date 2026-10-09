"""
test_report_agent.py

Tests for evidence-based institutional research report generation.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from agents.report import report_node
from models.report_models import InstitutionalResearchReport


def _report() -> InstitutionalResearchReport:
    """Create a valid report fixture for deterministic agent tests."""
    return InstitutionalResearchReport(
        executive_summary="The supplied ratios indicate positive revenue growth.",
        business_and_news="No business or news evidence was supplied.",
        financial_analysis="Revenue growth was 10% in the supplied ratio data.",
        valuation_analysis="No DCF valuation was supplied.",
        risk_analysis="No structured risk analysis was supplied.",
        catalysts=[],
        overall_view="balanced",
        conclusion="The available evidence is limited to one financial metric.",
        evidence=[
            {
                "claim": "Revenue growth was 10%.",
                "source": "ratio_analysis",
                "supporting_evidence": "revenue_growth = 10.0",
            }
        ],
        data_gaps=["No DCF valuation was supplied."],
    )


def test_report_node_synthesizes_and_cites_supplied_evidence() -> None:
    """Verify the report uses supplied evidence and persists structured output."""
    state = {
        "messages": [HumanMessage(content="Write a full research report.")],
        "company": "Example Co",
        "task": "report",
        "news": "",
        "final_report": "",
        "ratio_analysis": {"revenue_growth": 10.0},
    }
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = _report()

    with patch("agents.report.llm") as mock_llm:
        mock_llm.with_structured_output.return_value = structured_llm
        updated_state = report_node(state)  # type: ignore[arg-type]

    mock_llm.with_structured_output.assert_called_once_with(
        InstitutionalResearchReport
    )
    request = json.loads(structured_llm.invoke.call_args.args[0][1][1])
    assert request["company"] == "Example Co"
    assert request["question"] == "Write a full research report."
    assert request["evidence"] == {
        "ratio_analysis": {"revenue_growth": 10.0}
    }
    assert updated_state["institutional_report"]["overall_view"] == "balanced"
    assert "Institutional Equity Research Report — Example Co" in updated_state[
        "final_report"
    ]
    assert "source: ratio_analysis" in updated_state["final_report"]


def test_report_node_returns_gaps_when_evidence_is_missing() -> None:
    """Verify a report with no inputs stays local and does not call an LLM."""
    state = {
        "messages": [HumanMessage(content="Write a report on Example Co.")],
        "company": "Example Co",
        "task": "report",
        "news": "",
        "final_report": "",
    }

    with patch("agents.report.llm") as mock_llm:
        updated_state = report_node(state)  # type: ignore[arg-type]

    mock_llm.with_structured_output.assert_not_called()
    report = updated_state["institutional_report"]
    assert report["overall_view"] == "insufficient_evidence"
    assert len(report["data_gaps"]) > 0
    assert "cannot be completed" in updated_state["final_report"]


def test_report_node_rejects_citations_to_unsupplied_sources() -> None:
    """Verify the report cannot cite a source absent from graph state."""
    state = {
        "messages": [],
        "company": "Example Co",
        "task": "report",
        "news": "",
        "final_report": "",
        "ratio_analysis": {"revenue_growth": 10.0},
    }
    invalid_report = _report().model_dump() | {
        "evidence": [
            {
                "claim": "A DCF result exists.",
                "source": "dcf_analysis",
                "supporting_evidence": "Unsupported citation.",
            }
        ]
    }
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = invalid_report

    with patch("agents.report.llm") as mock_llm:
        mock_llm.with_structured_output.return_value = structured_llm
        with pytest.raises(ValueError, match="not supplied"):
            report_node(state)  # type: ignore[arg-type]


def test_news_report_remains_available() -> None:
    """Verify the existing news-to-report workflow remains compatible."""
    state = {
        "messages": [],
        "company": "Example Co",
        "task": "news",
        "news": "Supplied news result.",
        "final_report": "",
    }

    updated_state = report_node(state)  # type: ignore[arg-type]

    assert "Latest News Report" in updated_state["final_report"]
    assert "Supplied news result." in updated_state["final_report"]
