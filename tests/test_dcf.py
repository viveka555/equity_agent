"""
test_dcf.py

Tests for the DCF calculation engine.
"""

import pytest

from models.dcf_models import (
    DCFAssumptions,
    DCFForecastInput,
)
from valuation.dcf import (
    calculate_discounted_fcfs,
    calculate_free_cash_flow,
    calculate_projected_revenue,
    discount_cash_flow,
)


def test_projected_revenue() -> None:
    """Verify multi-year revenue forecasting."""

    revenue = calculate_projected_revenue(
        current_revenue=1000.0,
        revenue_growth=10.0,
        forecast_years=3,
    )

    assert revenue == pytest.approx(
        [1100.0, 1210.0, 1331.0]
    )


def test_free_cash_flow() -> None:
    """Verify FCF calculation."""

    fcf = calculate_free_cash_flow(
        revenue=1000.0,
        ebit_margin=20.0,
        tax_rate=25.0,
        depreciation=50.0,
        capital_expenditure=80.0,
        change_in_working_capital=20.0,
    )

    # EBIT = 200
    # NOPAT = 150
    # FCF = 150 + 50 - 80 - 20 = 100
    assert fcf == pytest.approx(100.0)


def test_discount_cash_flow() -> None:
    """Verify present-value calculation."""

    present_value = discount_cash_flow(
        cash_flow=121.0,
        discount_rate=10.0,
        period=2,
    )

    # 121 / (1.10)^2 = 100
    assert present_value == pytest.approx(100.0)


def test_discount_cash_flow_rejects_invalid_period() -> None:
    """Verify invalid discount periods are rejected."""

    with pytest.raises(ValueError):
        discount_cash_flow(
            cash_flow=100.0,
            discount_rate=10.0,
            period=0,
        )


def test_discounted_fcfs() -> None:
    """Verify the complete forecast-to-PV calculation."""

    forecast_input = DCFForecastInput(
        current_revenue=1000.0,
        current_ebit=200.0,
        depreciation=50.0,
        capital_expenditure=80.0,
        change_in_working_capital=20.0,
        total_debt=100.0,
        cash_and_equivalents=50.0,
        shares_outstanding=100.0,
    )

    assumptions = DCFAssumptions(
        revenue_growth=10.0,
        ebit_margin=20.0,
        tax_rate=25.0,
        wacc=10.0,
        terminal_growth_rate=4.0,
        forecast_years=2,
    )

    discounted_fcfs = calculate_discounted_fcfs(
        forecast_input=forecast_input,
        assumptions=assumptions,
    )

    assert len(discounted_fcfs) == 2

    # Year 1:
    # Revenue = 1100
    # EBIT = 220
    # NOPAT = 165
    # FCF = 165 + 50 - 80 - 20 = 115
    # PV = 115 / 1.10
    assert discounted_fcfs[0] == pytest.approx(
        115 / 1.10
    )

    # Year 2:
    # Revenue = 1210
    # EBIT = 242
    # NOPAT = 181.5
    # FCF = 181.5 + 50 - 80 - 20 = 131.5
    # PV = 131.5 / 1.10^2
    assert discounted_fcfs[1] == pytest.approx(
        131.5 / (1.10 ** 2)
    )