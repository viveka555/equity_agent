"""
wacc.py

Weighted Average Cost of Capital (WACC) calculation engine.

Responsibilities
----------------
- Calculate cost of equity using CAPM.
- Calculate after-tax cost of debt.
- Calculate capital structure weights.
- Calculate WACC.

Architecture Role
-----------------
Market / Financial Inputs
        ↓
Cost of Equity
        ↓
Cost of Debt
        ↓
Capital Structure
        ↓
WACC
        ↓
DCF Engine
"""

from __future__ import annotations


def calculate_cost_of_equity(
    risk_free_rate: float,
    beta: float,
    market_return: float,
) -> float:
    """
    Calculate cost of equity using CAPM.

    Formula:

        Ke = Rf + Beta × (Rm - Rf)

    Args:
        risk_free_rate: Risk-free rate percentage.
        beta: Company's equity beta.
        market_return: Expected market return percentage.

    Returns:
        Cost of equity as a percentage.

    Raises:
        ValueError: If beta is negative.
    """
    if beta < 0:
        raise ValueError("Beta cannot be negative.")

    return risk_free_rate + (
        beta * (market_return - risk_free_rate)
    )


def calculate_after_tax_cost_of_debt(
    cost_of_debt: float,
    tax_rate: float,
) -> float:
    """
    Calculate the after-tax cost of debt.

    Formula:

        After-tax Kd = Kd × (1 - Tax Rate)

    Args:
        cost_of_debt: Pre-tax cost of debt percentage.
        tax_rate: Corporate tax rate percentage.

    Returns:
        After-tax cost of debt as a percentage.

    Raises:
        ValueError: If tax rate is outside 0-100%.
    """
    if not 0 <= tax_rate < 100:
        raise ValueError(
            "Tax rate must be between 0% and 100%."
        )

    return cost_of_debt * (1 - tax_rate / 100)


def calculate_wacc(
    market_value_equity: float,
    market_value_debt: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float,
) -> float:
    """
    Calculate Weighted Average Cost of Capital.

    Formula:

        WACC =
            Equity Weight × Cost of Equity
            +
            Debt Weight × After-tax Cost of Debt

    Args:
        market_value_equity: Market value of equity.
        market_value_debt: Market value of debt.
        cost_of_equity: Cost of equity percentage.
        cost_of_debt: Pre-tax cost of debt percentage.
        tax_rate: Corporate tax rate percentage.

    Returns:
        WACC as a percentage.

    Raises:
        ValueError: If capital values are invalid or total capital
            is zero.
    """
    if market_value_equity < 0:
        raise ValueError(
            "Market value of equity cannot be negative."
        )

    if market_value_debt < 0:
        raise ValueError(
            "Market value of debt cannot be negative."
        )

    total_capital = market_value_equity + market_value_debt

    if total_capital <= 0:
        raise ValueError(
            "Total capital must be greater than zero."
        )

    after_tax_cost_of_debt = calculate_after_tax_cost_of_debt(
        cost_of_debt=cost_of_debt,
        tax_rate=tax_rate,
    )

    equity_weight = market_value_equity / total_capital
    debt_weight = market_value_debt / total_capital

    wacc = (
        equity_weight * cost_of_equity
        + debt_weight * after_tax_cost_of_debt
    )

    return wacc