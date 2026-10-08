"""
test_valuation.py

Tests for the DCF valuation bridge.
"""

import pytest

from valuation.valuation import (
    calculate_enterprise_value,
    calculate_equity_value,
    calculate_intrinsic_value_per_share,
    calculate_present_value_terminal_value,
    calculate_terminal_value,
)


def test_calculate_terminal_value() -> None:
    """Verify Gordon Growth terminal value."""

    result = calculate_terminal_value(
        final_year_fcf=131.5,
        wacc=10.0,
        terminal_growth_rate=4.0,
    )

    # Next FCF = 131.5 × 1.04 = 136.76
    # TV = 136.76 / (0.10 - 0.04)
    assert result == pytest.approx(
        136.76 / 0.06
    )


def test_wacc_must_exceed_terminal_growth() -> None:
    """Verify invalid WACC/growth relationship."""

    with pytest.raises(ValueError):
        calculate_terminal_value(
            final_year_fcf=131.5,
            wacc=4.0,
            terminal_growth_rate=4.0,
        )


def test_present_value_terminal_value() -> None:
    """Verify terminal value discounting."""

    result = calculate_present_value_terminal_value(
        terminal_value=1000.0,
        wacc=10.0,
        forecast_years=2,
    )

    assert result == pytest.approx(
        1000.0 / (1.10 ** 2)
    )


def test_enterprise_value() -> None:
    """Verify enterprise value calculation."""

    result = calculate_enterprise_value(
        present_value_of_forecast=5000.0,
        present_value_of_terminal_value=7000.0,
    )

    assert result == pytest.approx(12000.0)


def test_equity_value() -> None:
    """Verify enterprise-to-equity value bridge."""

    result = calculate_equity_value(
        enterprise_value=12000.0,
        total_debt=3000.0,
        cash_and_equivalents=1000.0,
    )

    # 12000 - 3000 + 1000 = 10000
    assert result == pytest.approx(10000.0)


def test_intrinsic_value_per_share() -> None:
    """Verify intrinsic value per share."""

    result = calculate_intrinsic_value_per_share(
        equity_value=10000.0,
        shares_outstanding=100.0,
    )

    assert result == pytest.approx(100.0)


def test_zero_shares_are_rejected() -> None:
    """Verify invalid share count is rejected."""

    with pytest.raises(ValueError):
        calculate_intrinsic_value_per_share(
            equity_value=10000.0,
            shares_outstanding=0.0,
        )