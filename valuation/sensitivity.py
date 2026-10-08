"""
sensitivity.py

DCF sensitivity analysis engine.

Responsibilities
----------------
- Calculate intrinsic value across different WACC assumptions.
- Calculate intrinsic value across different terminal growth rates.
- Generate a WACC × terminal-growth sensitivity matrix.

Architecture Role
-----------------
DCF Valuation
      ↓
Sensitivity Engine
      ↓
WACC × Terminal Growth
      ↓
Intrinsic Value Matrix
"""

from __future__ import annotations

from valuation.valuation import (
    calculate_equity_value,
    calculate_intrinsic_value_per_share,
    calculate_present_value_terminal_value,
    calculate_terminal_value,
)


def calculate_sensitivity_value(
    present_value_of_forecast: float,
    final_year_fcf: float,
    total_debt: float,
    cash_and_equivalents: float,
    shares_outstanding: float,
    wacc: float,
    terminal_growth_rate: float,
    forecast_years: int,
) -> float:
    """
    Calculate intrinsic value per share for one WACC/growth scenario.

    Args:
        present_value_of_forecast: Present value of forecast-period
            FCFs.
        final_year_fcf: FCF in the final forecast year.
        total_debt: Total debt.
        cash_and_equivalents: Cash and equivalents.
        shares_outstanding: Number of shares outstanding.
        wacc: Scenario WACC percentage.
        terminal_growth_rate: Scenario terminal growth percentage.
        forecast_years: Number of forecast years.

    Returns:
        Intrinsic value per share.
    """

    terminal_value = calculate_terminal_value(
        final_year_fcf=final_year_fcf,
        wacc=wacc,
        terminal_growth_rate=terminal_growth_rate,
    )

    present_value_terminal = (
        calculate_present_value_terminal_value(
            terminal_value=terminal_value,
            wacc=wacc,
            forecast_years=forecast_years,
        )
    )

    enterprise_value = present_value_of_forecast + (
        present_value_terminal
    )

    equity_value = calculate_equity_value(
        enterprise_value=enterprise_value,
        total_debt=total_debt,
        cash_and_equivalents=cash_and_equivalents,
    )

    return calculate_intrinsic_value_per_share(
        equity_value=equity_value,
        shares_outstanding=shares_outstanding,
    )


def generate_sensitivity_matrix(
    present_value_of_forecast: float,
    final_year_fcf: float,
    total_debt: float,
    cash_and_equivalents: float,
    shares_outstanding: float,
    wacc_values: list[float],
    terminal_growth_values: list[float],
    forecast_years: int,
) -> dict[float, dict[float, float]]:
    """
    Generate a WACC × terminal-growth sensitivity matrix.

    Args:
        present_value_of_forecast: Present value of forecast FCFs.
        final_year_fcf: Final forecast-year FCF.
        total_debt: Total debt.
        cash_and_equivalents: Cash and equivalents.
        shares_outstanding: Number of shares.
        wacc_values: WACC scenarios.
        terminal_growth_values: Terminal growth scenarios.
        forecast_years: Number of forecast years.

    Returns:
        Nested dictionary where:
            outer key = WACC
            inner key = terminal growth rate
            value = intrinsic value per share.
    """

    sensitivity_matrix: dict[float, dict[float, float]] = {}

    for wacc in wacc_values:
        sensitivity_matrix[wacc] = {}

        for terminal_growth in terminal_growth_values:
            sensitivity_matrix[wacc][terminal_growth] = (
                calculate_sensitivity_value(
                    present_value_of_forecast=(
                        present_value_of_forecast
                    ),
                    final_year_fcf=final_year_fcf,
                    total_debt=total_debt,
                    cash_and_equivalents=cash_and_equivalents,
                    shares_outstanding=shares_outstanding,
                    wacc=wacc,
                    terminal_growth_rate=terminal_growth,
                    forecast_years=forecast_years,
                )
            )

    return sensitivity_matrix