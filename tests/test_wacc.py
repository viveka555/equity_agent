"""
test_wacc.py

Tests for the WACC calculation engine.
"""

import pytest

from valuation.wacc import (
    calculate_after_tax_cost_of_debt,
    calculate_cost_of_equity,
    calculate_wacc,
)


def test_calculate_cost_of_equity() -> None:
    """Verify CAPM cost-of-equity calculation."""

    result = calculate_cost_of_equity(
        risk_free_rate=7.0,
        beta=1.2,
        market_return=12.0,
    )

    # 7 + 1.2 × (12 - 7) = 13
    assert result == pytest.approx(13.0)


def test_negative_beta_is_rejected() -> None:
    """Verify negative beta is rejected."""

    with pytest.raises(ValueError):
        calculate_cost_of_equity(
            risk_free_rate=7.0,
            beta=-1.0,
            market_return=12.0,
        )


def test_after_tax_cost_of_debt() -> None:
    """Verify after-tax cost of debt."""

    result = calculate_after_tax_cost_of_debt(
        cost_of_debt=8.0,
        tax_rate=25.0,
    )

    # 8 × (1 - 0.25) = 6
    assert result == pytest.approx(6.0)


def test_invalid_tax_rate_is_rejected() -> None:
    """Verify invalid tax rates are rejected."""

    with pytest.raises(ValueError):
        calculate_after_tax_cost_of_debt(
            cost_of_debt=8.0,
            tax_rate=100.0,
        )


def test_calculate_wacc() -> None:
    """Verify complete WACC calculation."""

    result = calculate_wacc(
        market_value_equity=8000.0,
        market_value_debt=2000.0,
        cost_of_equity=13.0,
        cost_of_debt=8.0,
        tax_rate=25.0,
    )

    # Equity weight = 80%
    # Debt weight = 20%
    # After-tax debt cost = 6%
    # WACC = 0.8 × 13 + 0.2 × 6 = 11.6%
    assert result == pytest.approx(11.6)


def test_zero_total_capital_is_rejected() -> None:
    """Verify zero total capital is rejected."""

    with pytest.raises(ValueError):
        calculate_wacc(
            market_value_equity=0.0,
            market_value_debt=0.0,
            cost_of_equity=13.0,
            cost_of_debt=8.0,
            tax_rate=25.0,
        )