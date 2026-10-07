"""
test_ratio_models.py

Tests for the Financial Ratio Agent data models.

Responsibilities
----------------
- Verify valid financial input data.
- Verify calculated ratio model creation.
- Verify validation rejects invalid financial values.
"""

import pytest
from pydantic import ValidationError

from models.ratio_models import FinancialInput, RatioAnalysis


def test_financial_input() -> None:
    """
    Verify that valid financial data creates a FinancialInput model.
    """
    financial_data = FinancialInput(
        revenue=1000.0,
        previous_revenue=900.0,
        ebitda=200.0,
        ebit=150.0,
        net_profit=100.0,
        total_assets=2000.0,
        total_equity=1200.0,
        total_debt=400.0,
        current_assets=600.0,
        current_liabilities=300.0,
        interest_expense=25.0,
        eps=10.0,
        previous_eps=8.0,
    )

    assert financial_data.revenue == 1000.0
    assert financial_data.net_profit == 100.0
    assert financial_data.eps == 10.0


def test_financial_input_rejects_negative_values() -> None:
    """
    Verify that negative financial values are rejected.
    """
    with pytest.raises(ValidationError):
        FinancialInput(
            revenue=-1000.0,
            previous_revenue=900.0,
            ebitda=200.0,
            ebit=150.0,
            net_profit=100.0,
            total_assets=2000.0,
            total_equity=1200.0,
            total_debt=400.0,
            current_assets=600.0,
            current_liabilities=300.0,
            interest_expense=25.0,
            eps=10.0,
            previous_eps=8.0,
        )


def test_ratio_analysis() -> None:
    """
    Verify that calculated ratios can be represented correctly.
    """
    ratios = RatioAnalysis(
        revenue_growth=11.11,
        ebitda_margin=20.0,
        net_profit_margin=10.0,
        return_on_equity=8.33,
        return_on_assets=5.0,
        debt_to_equity=0.33,
        current_ratio=2.0,
        interest_coverage=6.0,
        eps_growth=25.0,
    )

    assert ratios.revenue_growth == 11.11
    assert ratios.ebitda_margin == 20.0
    assert ratios.debt_to_equity == 0.33
    