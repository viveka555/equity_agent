"""
dcf_agent.py

Discounted Cash Flow Agent for the Institutional Equity Research System.

The agent accepts validated financial data and assumptions through graph
state, delegates valuation math to the valuation engines, and stores a
serializable analysis in the shared state.
"""

from __future__ import annotations

import logging
from typing import Any

from models.dcf_models import (
    DCFAssumptions,
    DCFForecastInput,
    DCFValuationResult,
)
from state import GraphState
from valuation.dcf import (
    calculate_discounted_fcfs,
    calculate_free_cash_flow,
    calculate_projected_revenue,
)
from valuation.sensitivity import generate_sensitivity_matrix
from valuation.valuation import (
    calculate_enterprise_value,
    calculate_equity_value,
    calculate_intrinsic_value_per_share,
    calculate_present_value_terminal_value,
    calculate_terminal_value,
)
from valuation.wacc import (
    calculate_cost_of_equity,
    calculate_wacc,
)

logger = logging.getLogger(__name__)


def _calculate_assumptions(state: GraphState) -> DCFAssumptions:
    """Validate assumptions and calculate WACC when market inputs exist.

    Args:
        state: Shared graph state containing DCF assumptions and optional
            market inputs for WACC calculation.

    Returns:
        Validated assumptions with calculated WACC when applicable.

    Raises:
        ValueError: If DCF assumptions are missing.
        KeyError: If supplied WACC inputs are incomplete.
    """
    raw_assumptions = state.get("dcf_assumptions")
    if not raw_assumptions:
        raise ValueError("No DCF assumptions are available.")

    assumptions = DCFAssumptions.model_validate(raw_assumptions)
    wacc_inputs = state.get("dcf_wacc_inputs")
    if not wacc_inputs:
        return assumptions

    cost_of_equity = calculate_cost_of_equity(
        risk_free_rate=wacc_inputs["risk_free_rate"],
        beta=wacc_inputs["beta"],
        market_return=wacc_inputs["market_return"],
    )
    wacc = calculate_wacc(
        market_value_equity=wacc_inputs["market_value_equity"],
        market_value_debt=wacc_inputs["market_value_debt"],
        cost_of_equity=cost_of_equity,
        cost_of_debt=wacc_inputs["cost_of_debt"],
        tax_rate=assumptions.tax_rate,
    )
    return DCFAssumptions.model_validate(
        assumptions.model_dump() | {"wacc": wacc}
    )


def dcf_node(state: GraphState) -> GraphState:
    """Run a complete DCF valuation and update graph state.

    Required state inputs are ``dcf_forecast_input`` and
    ``dcf_assumptions``. Optional ``dcf_wacc_inputs`` calculates WACC from
    market data; optional ``dcf_sensitivity_inputs`` supplies scenario
    lists for sensitivity analysis.

    Args:
        state: Shared LangGraph state.

    Returns:
        Updated graph state with a serializable ``dcf_analysis`` result.

    Raises:
        ValueError: If required financial inputs are unavailable.
        pydantic.ValidationError: If inputs do not match their data models.
        KeyError: If provided WACC inputs are incomplete.
    """
    company = state.get("company", "unknown")
    logger.info("Running DCF Agent for %s", company)

    raw_forecast = state.get("dcf_forecast_input")
    if not raw_forecast:
        raise ValueError(
            f"No DCF financial inputs available for company: {company}"
        )

    forecast_input = DCFForecastInput.model_validate(raw_forecast)
    assumptions = _calculate_assumptions(state)

    projected_revenue = calculate_projected_revenue(
        current_revenue=forecast_input.current_revenue,
        revenue_growth=assumptions.revenue_growth,
        forecast_years=assumptions.forecast_years,
    )
    discounted_fcfs = calculate_discounted_fcfs(
        forecast_input=forecast_input,
        assumptions=assumptions,
    )
    final_year_fcf = calculate_free_cash_flow(
        revenue=projected_revenue[-1],
        ebit_margin=assumptions.ebit_margin,
        tax_rate=assumptions.tax_rate,
        depreciation=forecast_input.depreciation,
        capital_expenditure=forecast_input.capital_expenditure,
        change_in_working_capital=forecast_input.change_in_working_capital,
    )

    terminal_value = calculate_terminal_value(
        final_year_fcf=final_year_fcf,
        wacc=assumptions.wacc,
        terminal_growth_rate=assumptions.terminal_growth_rate,
    )
    present_value_of_forecast = sum(discounted_fcfs)
    present_value_of_terminal_value = calculate_present_value_terminal_value(
        terminal_value=terminal_value,
        wacc=assumptions.wacc,
        forecast_years=assumptions.forecast_years,
    )
    enterprise_value = calculate_enterprise_value(
        present_value_of_forecast=present_value_of_forecast,
        present_value_of_terminal_value=present_value_of_terminal_value,
    )
    equity_value = calculate_equity_value(
        enterprise_value=enterprise_value,
        total_debt=forecast_input.total_debt,
        cash_and_equivalents=forecast_input.cash_and_equivalents,
    )
    intrinsic_value_per_share = calculate_intrinsic_value_per_share(
        equity_value=equity_value,
        shares_outstanding=forecast_input.shares_outstanding,
    )

    valuation = DCFValuationResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
        terminal_value=terminal_value,
        present_value_of_forecast=present_value_of_forecast,
        present_value_of_terminal_value=present_value_of_terminal_value,
    )
    analysis: dict[str, Any] = {
        "assumptions": assumptions.model_dump(),
        "projected_revenue": projected_revenue,
        "discounted_fcfs": discounted_fcfs,
        "final_year_fcf": final_year_fcf,
        "valuation": valuation.model_dump(),
    }
    report = (
        f"DCF Valuation — {company}\n\n"
        f"Intrinsic value per share: {intrinsic_value_per_share:,.2f}\n"
        f"Enterprise value: {enterprise_value:,.2f}\n"
        f"Equity value: {equity_value:,.2f}\n"
        f"WACC: {assumptions.wacc:.2f}%\n"
        f"Terminal growth rate: {assumptions.terminal_growth_rate:.2f}%\n"
        "Values use the same monetary unit as the supplied financial data."
    )

    sensitivity_inputs = state.get("dcf_sensitivity_inputs")
    if sensitivity_inputs:
        analysis["sensitivity_matrix"] = generate_sensitivity_matrix(
            present_value_of_forecast=present_value_of_forecast,
            final_year_fcf=final_year_fcf,
            total_debt=forecast_input.total_debt,
            cash_and_equivalents=forecast_input.cash_and_equivalents,
            shares_outstanding=forecast_input.shares_outstanding,
            wacc_values=sensitivity_inputs["wacc_values"],
            terminal_growth_values=(
                sensitivity_inputs["terminal_growth_values"]
            ),
            forecast_years=assumptions.forecast_years,
        )

    logger.info("DCF analysis completed for %s", company)
    return {**state, "dcf_analysis": analysis, "final_report": report}
