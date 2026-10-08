"""
valuation.py

DCF valuation bridge for the Institutional Equity Research Agent.

Responsibilities
----------------
- Calculate terminal value using the Gordon Growth Model.
- Discount terminal value to present value.
- Calculate enterprise value.
- Calculate equity value.
- Calculate intrinsic value per share.

Architecture Role
-----------------
Projected FCFs
      ↓
PV of Forecast FCFs
      +
Terminal Value
      ↓
PV of Terminal Value
      ↓
Enterprise Value
      ↓
- Debt
+ Cash
      ↓
Equity Value
      ↓
Intrinsic Value / Share
"""

from __future__ import annotations


def calculate_terminal_value(
    final_year_fcf: float,
    wacc: float,
    terminal_growth_rate: float,
) -> float:
    """
    Calculate terminal value using the Gordon Growth Model.

    Formula:

        Terminal Value =
            FCF_(n+1) / (WACC - g)

        FCF_(n+1) =
            FCF_n × (1 + g)

    Args:
        final_year_fcf: Free cash flow in the final forecast year.
        wacc: Weighted Average Cost of Capital percentage.
        terminal_growth_rate: Perpetual growth rate percentage.

    Returns:
        Terminal value.

    Raises:
        ValueError: If WACC is less than or equal to terminal growth.
    """
    if wacc <= terminal_growth_rate:
        raise ValueError(
            "WACC must be greater than the terminal growth rate."
        )

    growth_rate = terminal_growth_rate / 100
    discount_rate = wacc / 100

    next_year_fcf = final_year_fcf * (1 + growth_rate)

    return next_year_fcf / (discount_rate - growth_rate)


def calculate_present_value_terminal_value(
    terminal_value: float,
    wacc: float,
    forecast_years: int,
) -> float:
    """
    Discount terminal value to its present value.

    Formula:

        PV(TV) = TV / (1 + WACC)^n

    Args:
        terminal_value: Terminal value.
        wacc: WACC percentage.
        forecast_years: Number of forecast years.

    Returns:
        Present value of terminal value.
    """
    if forecast_years <= 0:
        raise ValueError(
            "Forecast years must be greater than zero."
        )

    discount_rate = wacc / 100

    return terminal_value / (
        (1 + discount_rate) ** forecast_years
    )


def calculate_enterprise_value(
    present_value_of_forecast: float,
    present_value_of_terminal_value: float,
) -> float:
    """
    Calculate enterprise value.

    Formula:

        EV = PV(Forecast FCFs) + PV(Terminal Value)

    Args:
        present_value_of_forecast: Present value of forecast-period
            free cash flows.
        present_value_of_terminal_value: Present value of terminal
            value.

    Returns:
        Enterprise value.
    """
    return (
        present_value_of_forecast
        + present_value_of_terminal_value
    )


def calculate_equity_value(
    enterprise_value: float,
    total_debt: float,
    cash_and_equivalents: float,
) -> float:
    """
    Calculate equity value from enterprise value.

    Formula:

        Equity Value = EV - Debt + Cash

    Args:
        enterprise_value: Enterprise value.
        total_debt: Total debt.
        cash_and_equivalents: Cash and cash equivalents.

    Returns:
        Equity value.
    """
    return (
        enterprise_value
        - total_debt
        + cash_and_equivalents
    )


def calculate_intrinsic_value_per_share(
    equity_value: float,
    shares_outstanding: float,
) -> float:
    """
    Calculate intrinsic value per share.

    Formula:

        Intrinsic Value / Share =
            Equity Value / Shares Outstanding

    Args:
        equity_value: Total equity value.
        shares_outstanding: Number of shares outstanding.

    Returns:
        Intrinsic value per share.

    Raises:
        ValueError: If shares outstanding is zero or negative.
    """
    if shares_outstanding <= 0:
        raise ValueError(
            "Shares outstanding must be greater than zero."
        )

    return equity_value / shares_outstanding