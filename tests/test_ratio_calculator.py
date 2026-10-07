"""
test_ratio_calculator.py

Tests for the financial ratio calculation engine.

Responsibilities
----------------
- Verify financial ratios are calculated correctly.
- Verify zero denominators are handled safely.
"""

import pytest

from models.ratio_models import FinancialInput
from tools.ratio_calculator import calculate_ratios


def _create_financial_data() -> FinancialInput:
    """
    Create deterministic financial data for ratio tests.

    Returns:
        Valid FinancialInput test fixture.
    """
    return FinancialInput(
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


def test_calculate_ratios() -> None:
    """
    Verify key financial ratios are calculated correctly.
    """
    financial_data = _create_financial_data()

    ratios = calculate_ratios(financial_data)

    assert round(ratios.revenue_growth, 2) == 11.11
    assert ratios.ebitda_margin == 20.0
    assert ratios.net_profit_margin == 10.0
    assert round(ratios.return_on_equity, 2) == 8.33
    assert ratios.return_on_assets == 5.0
    assert round(ratios.debt_to_equity, 2) == 0.33
    assert ratios.current_ratio == 2.0
    assert ratios.interest_coverage == 6.0
    assert ratios.eps_growth == 25.0


def test_zero_denominator() -> None:
    """
    Verify zero denominators raise a meaningful ValueError.
    """
    financial_data = _create_financial_data()

    financial_data = financial_data.model_copy(
        update={"total_equity": 0.0}
    )

    with pytest.raises(
        ValueError,
        match="denominator is zero",
    ):
        calculate_ratios(financial_data)