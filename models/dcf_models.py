"""
dcf_models.py

Pydantic data models for the Discounted Cash Flow (DCF) valuation
engine of the Institutional Equity Research Agent.

Responsibilities
----------------
- Define DCF assumptions.
- Define financial forecast inputs.
- Define DCF valuation outputs.
- Validate financial values before they reach the calculation engine.

Architecture Role
-----------------
Financial Data
      ↓
DCF Models
      ↓
DCF Calculation Engine
      ↓
Valuation Output
      ↓
DCF Agent
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DCFAssumptions(BaseModel):
    """
    Core assumptions used by the DCF valuation engine.

    All percentage values are represented as percentages rather than
    decimal fractions.

    Example:
        12.0 represents 12%.
    """

    revenue_growth: float = Field(
        ...,
        description="Expected annual revenue growth percentage.",
    )

    ebit_margin: float = Field(
        ...,
        description="Expected EBIT margin percentage.",
    )

    tax_rate: float = Field(
        ...,
        ge=0,
        lt=100,
        description="Corporate tax rate percentage.",
    )

    wacc: float = Field(
        ..., # <--- Marks this field as REQUIRED
        gt=0,
        lt=100,
        description="Weighted Average Cost of Capital percentage.",
    )

    terminal_growth_rate: float = Field(
        ...,
        description="Terminal growth rate percentage.",
    )

    forecast_years: int = Field(
        ...,
        gt=0,
        le=20,
        description="Number of forecast years.",
    )


class DCFForecastInput(BaseModel):
    """
    Financial inputs required for DCF forecasting.

    Values are expressed in the same monetary unit, typically
    INR crore for an Indian company.
    """

    current_revenue: float = Field(
        ...,
        ge=0,
        description="Current annual revenue.",
    )

    current_ebit: float = Field(
        ...,
        ge=0,
        description="Current EBIT.",
    )

    depreciation: float = Field(
        ...,
        ge=0,
        description="Current depreciation and amortization.",
    )

    capital_expenditure: float = Field(
        ...,
        ge=0,
        description="Current capital expenditure.",
    )

    change_in_working_capital: float = Field(
        ...,
        description="Current change in working capital.",
    )

    total_debt: float = Field(
        ...,
        ge=0,
        description="Total interest-bearing debt.",
    )

    cash_and_equivalents: float = Field(
        ...,
        ge=0,
        description="Cash and cash equivalents.",
    )

    shares_outstanding: float = Field(
        ...,
        gt=0,
        description="Number of shares outstanding.",
    )


class DCFValuationResult(BaseModel):
    """
    Output produced by the DCF calculation engine.
    """

    enterprise_value: float = Field(
        ...,
        description="Calculated enterprise value.",
    )

    equity_value: float = Field(
        ...,
        description="Calculated equity value.",
    )

    intrinsic_value_per_share: float = Field(
        ...,
        description="Intrinsic value per share.",
    )

    terminal_value: float = Field(
        ...,
        description="Terminal value of the company.",
    )

    present_value_of_forecast: float = Field(
        ...,
        description="Present value of forecast-period free cash flows.",
    )

    present_value_of_terminal_value: float = Field(
        ...,
        description="Present value of terminal value.",
    )