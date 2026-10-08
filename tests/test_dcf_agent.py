"""
test_dcf_agent.py

Tests for DCF orchestration, WACC integration, and sensitivity analysis.
"""

import pytest

from agents.dcf_agent import dcf_node


def _dcf_state() -> dict[str, object]:
    """Create a complete DCF state using explicit test assumptions."""
    return {
        "messages": [],
        "company": "Example Co",
        "task": "dcf",
        "news": "",
        "final_report": "",
        "dcf_forecast_input": {
            "current_revenue": 1000.0,
            "current_ebit": 200.0,
            "depreciation": 50.0,
            "capital_expenditure": 80.0,
            "change_in_working_capital": 20.0,
            "total_debt": 100.0,
            "cash_and_equivalents": 50.0,
            "shares_outstanding": 100.0,
        },
        "dcf_assumptions": {
            "revenue_growth": 10.0,
            "ebit_margin": 20.0,
            "tax_rate": 25.0,
            "wacc": 10.0,
            "terminal_growth_rate": 4.0,
            "forecast_years": 2,
        },
    }


def test_dcf_node_calculates_valuation_and_sensitivity() -> None:
    """Verify DCF output includes the valuation and supplied scenarios."""
    state = _dcf_state()
    state["dcf_sensitivity_inputs"] = {
        "wacc_values": [9.0, 10.0],
        "terminal_growth_values": [3.0, 4.0],
    }

    updated_state = dcf_node(state)  # type: ignore[arg-type]

    analysis = updated_state["dcf_analysis"]
    assert analysis["projected_revenue"] == pytest.approx([1100.0, 1210.0])
    assert analysis["discounted_fcfs"] == pytest.approx(
        [115.0 / 1.1, 131.5 / (1.1**2)]
    )
    assert analysis["valuation"]["intrinsic_value_per_share"] > 0
    assert analysis["sensitivity_matrix"][9.0][3.0] > 0
    assert "DCF Valuation — Example Co" in updated_state["final_report"]


def test_dcf_node_calculates_wacc_from_market_inputs() -> None:
    """Verify supplied market data determines the DCF discount rate."""
    state = _dcf_state()
    state["dcf_wacc_inputs"] = {
        "risk_free_rate": 7.0,
        "beta": 1.2,
        "market_return": 12.0,
        "market_value_equity": 8000.0,
        "market_value_debt": 2000.0,
        "cost_of_debt": 8.0,
    }

    updated_state = dcf_node(state)  # type: ignore[arg-type]

    assert updated_state["dcf_analysis"]["assumptions"]["wacc"] == pytest.approx(
        11.6
    )


def test_dcf_node_requires_financial_inputs() -> None:
    """Verify missing financial data fails clearly."""
    state = _dcf_state()
    del state["dcf_forecast_input"]

    with pytest.raises(ValueError, match="No DCF financial inputs"):
        dcf_node(state)  # type: ignore[arg-type]


def test_dcf_node_requires_assumptions() -> None:
    """Verify missing DCF assumptions fail clearly."""
    state = _dcf_state()
    del state["dcf_assumptions"]

    with pytest.raises(ValueError, match="No DCF assumptions"):
        dcf_node(state)  # type: ignore[arg-type]
