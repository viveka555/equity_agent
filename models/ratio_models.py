"""
ratio_models.py

Pydantic data models for the Financial Ratio Agent.

Responsibilities
----------------
- Define the financial inputs required for ratio analysis.
- Define the calculated financial ratios returned by the agent.
- Provide validation and a stable data contract between
  financial-data tools, the Ratio Agent, and downstream agents.

Architecture Role
-----------------
Financial Data Tool
        ↓
FinancialInput
        ↓
Ratio Calculation
        ↓
RatioAnalysis
        ↓
Ratio Agent
        ↓
Institutional Research Report

Used by
--------
tools/financial_tools.py
agents/ratio_agent.py
valuation/
reports/
"""
from __future__ import annotations

from pydantic import BaseModel, Field

class FinancialInput(BaseModel):
    """
    Represent the raw financial data required for ratio analysis.

    Args:
        revenue: Total revenue for the financial period.
        previous_revenue: Revenue from the previous comparable period.
        ebitda: Earnings before interest, taxes, depreciation,
            and amortization.
        ebit: Earnings before interest and taxes.
        net_profit: Profit after tax.
        total_assets: Total assets.
        total_equity: Shareholders' equity.
        total_debt: Total interest-bearing debt.
        current_assets: Current assets.
        current_liabilities: Current liabilities.
        interest_expense: Interest expense for the period.
        eps: Earnings per share.
        previous_eps: EPS from the previous comparable period.

    Raises:
        ValueError: If a value that must be non-negative is negative.
    """
    revenue: float = Field(
        ...,
        ge=0,
        description="Total revenue for the financial period.",
    )

    previous_revenue: float = Field(
        ...,
        ge=0,
        description="Revenue from the previous comparable period.",
    )

    ebitda: float = Field(
        ...,
        ge=0,
        description="Earnings before interest, taxes, depreciation "
        "and amortization.",
    )

    ebit: float = Field(
    ...,
    ge=0,
    description="Earnings before interest and taxes.",
    )

    net_profit: float = Field(
        ...,
        ge=0,
        description="Profit after tax",
    )

    total_assets: float = Field(
        ...,
        ge=0,
        description="Total assets.",
    )

    total_equity: float = Field(
        ...,
        ge=0,
        description="Shareholders' equity.",
    )

    total_debt: float = Field(
        ...,
        ge=0,
        description="Total interest-bearing debt.",
    )

    current_assets: float = Field(
        ...,
        ge=0,
        description="Current assets.",
    )

    current_liabilities: float = Field(
        ...,
        ge=0,
        description="Current liabilities.",
    )

    interest_expense: float = Field(
        ...,
        ge=0,
        description="Interest expense for the financial period.",
    )

    eps: float = Field(
        ...,
        ge=0,
        description="Earnings per share.",
    )

    previous_eps: float = Field(
        ...,
        ge=0,
        description="EPS from the previous comparable period.",
    )


class RatioAnalysis(BaseModel):
    """
    Represent calculated financial ratios.

    All percentage-based ratios are expressed as percentages.

    Returns:
        A validated collection of financial ratios that can be
        consumed by the Ratio Agent and downstream components.
    """

    revenue_growth: float
    ebitda_margin: float
    net_profit_margin: float
    return_on_equity: float
    return_on_assets: float
    debt_to_equity: float
    current_ratio: float
    interest_coverage: float
    eps_growth: float
    