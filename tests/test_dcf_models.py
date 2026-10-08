"""
test_dcf_models.py

Tests for DCF Pydantic data models.
"""

import pytest
from pydantic import ValidationError

from models.dcf_models import (
    DCFAssumptions,
    DCFForecastInput,
    DCFValuationResult,
)


def test_dcf_assumptions() -> None:
    """Verify valid DCF assumptions are accepted."""

    assumptions = DCFAssumptions(
        revenue_growth=10.0,
        ebit_margin=18.0,
        tax_rate=25.0,
        wacc=10.0,
        terminal_growth_rate=4.0,
        forecast_years=5,
    )

    assert assumptions.revenue_growth == 10.0
    assert assumptions.wacc == 10.0
    assert assumptions.forecast_years == 5


def test_dcf_assumptions_reject_invalid_wacc() -> None:
    """Verify invalid WACC values are rejected."""

    with pytest.raises(ValidationError):
        DCFAssumptions(
            revenue_growth=10.0,
            ebit_margin=18.0,
            tax_rate=25.0,
            wacc=0.0,
            terminal_growth_rate=4.0,
            forecast_years=5,
        )


def test_dcf_forecast_input() -> None:
    """Verify valid DCF forecast inputs are accepted."""

    forecast = DCFForecastInput(
        current_revenue=10000.0,
        current_ebit=1800.0,
        depreciation=400.0,
        capital_expenditure=500.0,
        change_in_working_capital=200.0,
        total_debt=2500.0,
        cash_and_equivalents=1000.0,
        shares_outstanding=100.0,
    )

    assert forecast.current_revenue == 10000.0
    assert forecast.total_debt == 2500.0


def test_dcf_forecast_rejects_invalid_shares() -> None:
    """Verify zero shares outstanding are rejected."""

    with pytest.raises(ValidationError):
        DCFForecastInput(
            current_revenue=10000.0,
            current_ebit=1800.0,
            depreciation=400.0,
            capital_expenditure=500.0,
            change_in_working_capital=200.0,
            total_debt=2500.0,
            cash_and_equivalents=1000.0,
            shares_outstanding=0.0,
        )


def test_dcf_valuation_result() -> None:
    """Verify valid DCF valuation results are accepted."""

    result = DCFValuationResult(
        enterprise_value=50000.0,
        equity_value=48500.0,
        intrinsic_value_per_share=485.0,
        terminal_value=30000.0,
        present_value_of_forecast=20000.0,
        present_value_of_terminal_value=30000.0,
    )

    assert result.enterprise_value == 50000.0
    assert result.equity_value == 48500.0
    assert result.intrinsic_value_per_share == 485.0