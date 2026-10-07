"""
ratio_calculator.py

Financial ratio calculation engine for the Institutional Equity
Research Agent.

Responsibilities
----------------
- Calculate financial ratios from normalized financial data.
- Keep financial calculations independent from agent orchestration.
- Safely handle zero denominators.
- Return validated RatioAnalysis objects.

Architecture Role
-----------------
Financial Data Tool
        ↓
FinancialInput
        ↓
Ratio Calculator
        ↓
RatioAnalysis
        ↓
Ratio Agent
"""

from __future__ import annotations

import logging

from models.ratio_models import FinancialInput, RatioAnalysis

logger = logging.getLogger(__name__)


def _safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Divide two financial values safely.

    Args:
        numerator: Value being divided.
        denominator: Value used as the divisor.

    Returns:
        Division result.

    Raises:
        ValueError: If the denominator is zero.
    """
    if denominator == 0:
        raise ValueError(
            "Cannot calculate ratio because the denominator is zero."
        )

    return numerator / denominator


def calculate_ratios(
    financial_data: FinancialInput,
) -> RatioAnalysis:
    """
    Calculate key financial ratios.

    Percentage-based ratios are returned as percentages rather than
    decimal fractions.

    Args:
        financial_data: Validated financial statement data.

    Returns:
        Calculated and validated RatioAnalysis object.

    Raises:
        ValueError: If a required ratio has a zero denominator.
    """
    logger.info("Calculating financial ratios.")

    revenue_growth = (
        _safe_divide(
            financial_data.revenue - financial_data.previous_revenue,
            financial_data.previous_revenue,
        )
        * 100
    )

    ebitda_margin = (
        _safe_divide(
            financial_data.ebitda,
            financial_data.revenue,
        )
        * 100
    )

    net_profit_margin = (
        _safe_divide(
            financial_data.net_profit,
            financial_data.revenue,
        )
        * 100
    )

    return_on_equity = (
        _safe_divide(
            financial_data.net_profit,
            financial_data.total_equity,
        )
        * 100
    )

    return_on_assets = (
        _safe_divide(
            financial_data.net_profit,
            financial_data.total_assets,
        )
        * 100
    )

    debt_to_equity = _safe_divide(
        financial_data.total_debt,
        financial_data.total_equity,
    )

    current_ratio = _safe_divide(
        financial_data.current_assets,
        financial_data.current_liabilities,
    )

    interest_coverage = _safe_divide(
        financial_data.ebit,
        financial_data.interest_expense,
    )

    eps_growth = (
        _safe_divide(
            financial_data.eps - financial_data.previous_eps,
            financial_data.previous_eps,
        )
        * 100
    )

    return RatioAnalysis(
        revenue_growth=revenue_growth,
        ebitda_margin=ebitda_margin,
        net_profit_margin=net_profit_margin,
        return_on_equity=return_on_equity,
        return_on_assets=return_on_assets,
        debt_to_equity=debt_to_equity,
        current_ratio=current_ratio,
        interest_coverage=interest_coverage,
        eps_growth=eps_growth,
    )