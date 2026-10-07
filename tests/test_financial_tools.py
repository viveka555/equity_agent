"""
test_financial_tools.py

Tests for financial data normalization utilities.

Responsibilities
----------------
- Verify valid raw financial data can be normalized.
- Verify missing financial fields are rejected.
"""

import pytest

from models.ratio_models import FinancialInput
from tools.financial_tools import normalize_financial_data


def test_normalize_financial_data() -> None:
    """
    Verify valid raw financial data creates FinancialInput.
    """
    raw_data = {
        "revenue": 1000.0,
        "previous_revenue": 900.0,
        "ebitda": 200.0,
        "ebit": 150.0,
        "net_profit": 100.0,
        "total_assets": 2000.0,
        "total_equity": 1200.0,
        "total_debt": 400.0,
        "current_assets": 600.0,
        "current_liabilities": 300.0,
        "interest_expense": 25.0,
        "eps": 10.0,
        "previous_eps": 8.0,
    }

    financial_data = normalize_financial_data(raw_data)

    assert isinstance(financial_data, FinancialInput)
    assert financial_data.revenue == 1000.0
    assert financial_data.ebitda == 200.0
    assert financial_data.net_profit == 100.0


def test_missing_financial_field() -> None:
    """
    Verify missing required financial data raises ValueError.
    """
    incomplete_data = {
        "revenue": 1000.0,
        "previous_revenue": 900.0,
    }

    with pytest.raises(
        ValueError,
        match="Missing required financial fields",
    ):
        normalize_financial_data(incomplete_data)

        