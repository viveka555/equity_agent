"""
test_sensitivity.py

Tests for DCF sensitivity analysis.
"""

import pytest

from valuation.sensitivity import (
    calculate_sensitivity_value,
    generate_sensitivity_matrix,
)


def test_calculate_sensitivity_value() -> None:
    """Verify one WACC/growth valuation scenario."""

    result = calculate_sensitivity_value(
        present_value_of_forecast=5000.0,
        final_year_fcf=1000.0,
        total_debt=2000.0,
        cash_and_equivalents=500.0,
        shares_outstanding=100.0,
        wacc=10.0,
        terminal_growth_rate=4.0,
        forecast_years=5,
    )

    terminal_value = 1000.0 * 1.04 / (0.10 - 0.04)
    pv_terminal = terminal_value / (1.10 ** 5)

    expected_equity_value = (
        5000.0
        + pv_terminal
        - 2000.0
        + 500.0
    )

    expected_per_share = expected_equity_value / 100.0

    assert result == pytest.approx(expected_per_share)


def test_sensitivity_matrix_shape() -> None:
    """Verify all WACC/growth combinations are generated."""

    matrix = generate_sensitivity_matrix(
        present_value_of_forecast=5000.0,
        final_year_fcf=1000.0,
        total_debt=2000.0,
        cash_and_equivalents=500.0,
        shares_outstanding=100.0,
        wacc_values=[9.0, 10.0, 11.0],
        terminal_growth_values=[3.0, 4.0, 5.0],
        forecast_years=5,
    )

    assert len(matrix) == 3

    for wacc in [9.0, 10.0, 11.0]:
        assert len(matrix[wacc]) == 3


def test_lower_wacc_increases_valuation() -> None:
    """Verify lower WACC produces higher intrinsic value."""

    matrix = generate_sensitivity_matrix(
        present_value_of_forecast=5000.0,
        final_year_fcf=1000.0,
        total_debt=2000.0,
        cash_and_equivalents=500.0,
        shares_outstanding=100.0,
        wacc_values=[9.0, 10.0],
        terminal_growth_values=[4.0],
        forecast_years=5,
    )

    assert matrix[9.0][4.0] > matrix[10.0][4.0]


def test_higher_terminal_growth_increases_valuation() -> None:
    """Verify higher terminal growth produces higher valuation."""

    matrix = generate_sensitivity_matrix(
        present_value_of_forecast=5000.0,
        final_year_fcf=1000.0,
        total_debt=2000.0,
        cash_and_equivalents=500.0,
        shares_outstanding=100.0,
        wacc_values=[10.0],
        terminal_growth_values=[3.0, 4.0, 5.0],
        forecast_years=5,
    )

    assert matrix[10.0][5.0] > matrix[10.0][4.0]
    assert matrix[10.0][4.0] > matrix[10.0][3.0]


def test_invalid_wacc_growth_combination() -> None:
    """Verify WACC must exceed terminal growth."""

    with pytest.raises(ValueError):
        calculate_sensitivity_value(
            present_value_of_forecast=5000.0,
            final_year_fcf=1000.0,
            total_debt=2000.0,
            cash_and_equivalents=500.0,
            shares_outstanding=100.0,
            wacc=4.0,
            terminal_growth_rate=4.0,
            forecast_years=5,
        )