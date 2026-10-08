"""
dcf.py

Discounted Cash Flow (DCF) calculation engine for the
Institutional Equity Research Agent.

Responsibilities
----------------
- Forecast future revenue.
- Calculate EBIT and NOPAT.
- Calculate Free Cash Flow (FCF).
- Discount future FCFs to present value.
- Provide deterministic DCF calculation functions.

Architecture Role
-----------------
DCFForecastInput
        +
DCFAssumptions
        ↓
DCF Calculation Engine
        ↓
Projected FCFs
        ↓
Present Value of FCFs

WACC calculation, terminal value, and sensitivity analysis
are implemented in later DCF phases.
"""

from __future__ import annotations

from typing import List

from models.dcf_models import DCFAssumptions, DCFForecastInput


def calculate_projected_revenue(
    current_revenue: float,
    revenue_growth: float,
    forecast_years: int,
) -> List[float]:
    """
    Calculate projected revenue for each forecast year.

    Args:
        current_revenue: Current annual revenue.
        revenue_growth: Annual revenue growth percentage.
        forecast_years: Number of years to forecast.

    Returns:
        List containing projected revenue for each year.

    Raises:
        ValueError: If current revenue is negative or forecast years
            is not positive.
    """
    if current_revenue < 0:
        raise ValueError("Current revenue cannot be negative.")

    if forecast_years <= 0:
        raise ValueError("Forecast years must be greater than zero.")

    growth_factor = 1 + (revenue_growth / 100)

    projected_revenue: List[float] = []

    revenue = current_revenue

    for _ in range(forecast_years):
        revenue *= growth_factor
        projected_revenue.append(revenue)

    return projected_revenue


def calculate_free_cash_flow(
    revenue: float,
    ebit_margin: float,
    tax_rate: float,
    depreciation: float,
    capital_expenditure: float,
    change_in_working_capital: float,
) -> float:
    """
    Calculate Free Cash Flow for a forecast period.

    Formula:

        EBIT = Revenue × EBIT Margin

        NOPAT = EBIT × (1 - Tax Rate)

        FCF = NOPAT
              + Depreciation
              - Capital Expenditure
              - Change in Working Capital

    Args:
        revenue: Forecast revenue.
        ebit_margin: EBIT margin percentage.
        tax_rate: Tax rate percentage.
        depreciation: Depreciation and amortization.
        capital_expenditure: Capital expenditure.
        change_in_working_capital: Change in working capital.

    Returns:
        Free Cash Flow for the forecast period.
    """
    ebit = revenue * (ebit_margin / 100)

    nopat = ebit * (1 - (tax_rate / 100))

    free_cash_flow = (
        nopat
        + depreciation
        - capital_expenditure
        - change_in_working_capital
    )

    return free_cash_flow


def discount_cash_flow(
    cash_flow: float,
    discount_rate: float,
    period: int,
) -> float:
    """
    Discount a future cash flow to its present value.

    Formula:

        PV = FCF / (1 + WACC)^period

    Args:
        cash_flow: Future cash flow.
        discount_rate: Discount rate percentage.
        period: Number of years until the cash flow occurs.

    Returns:
        Present value of the cash flow.

    Raises:
        ValueError: If discount rate is invalid or period is invalid.
    """
    if discount_rate <= -100:
        raise ValueError(
            "Discount rate must be greater than -100%."
        )

    if period <= 0:
        raise ValueError("Discount period must be greater than zero.")

    discount_factor = (1 + discount_rate / 100) ** period

    return cash_flow / discount_factor


def calculate_discounted_fcfs(
    forecast_input: DCFForecastInput,
    assumptions: DCFAssumptions,
) -> list[float]:
    """
    Calculate the present value of forecast-period FCFs.

    Args:
        forecast_input: Current financial data.
        assumptions: DCF forecasting assumptions.

    Returns:
        List containing the present value of FCF for each
        forecast year.
    """
    projected_revenues = calculate_projected_revenue(
        current_revenue=forecast_input.current_revenue,
        revenue_growth=assumptions.revenue_growth,
        forecast_years=assumptions.forecast_years,
    )

    discounted_fcfs: list[float] = []

    for year, revenue in enumerate(
        projected_revenues,
        start=1,
    ):
        free_cash_flow = calculate_free_cash_flow(
            revenue=revenue,
            ebit_margin=assumptions.ebit_margin,
            tax_rate=assumptions.tax_rate,
            depreciation=forecast_input.depreciation,
            capital_expenditure=forecast_input.capital_expenditure,
            change_in_working_capital=(
                forecast_input.change_in_working_capital
            ),
        )

        present_value = discount_cash_flow(
            cash_flow=free_cash_flow,
            discount_rate=assumptions.wacc,
            period=year,
        )

        discounted_fcfs.append(present_value)

    return discounted_fcfs