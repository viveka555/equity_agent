"""
financial_tools.py

Financial data tools for the Institutional Equity Research Agent.

Responsibilities
----------------
- Provide normalized company financial data to downstream agents.
- Convert raw financial statement values into FinancialInput objects.
- Keep financial-data acquisition separate from ratio calculations.
- Provide a stable interface that can later be connected to an
  external financial-data provider.

Architecture Role
-----------------
External Financial Data Source
            ↓
Financial Data Tool
            ↓
FinancialInput
            ↓
Ratio Calculation Engine
            ↓
Ratio Agent

Used by
--------
agents/ratio_agent.py
valuation/dcf.py
reports/generator.py
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from models.ratio_models import FinancialInput

logger = logging.getLogger(__name__)


REQUIRED_FINANCIAL_FIELDS = {
    "revenue",
    "previous_revenue",
    "ebitda",
    "ebit",
    "net_profit",
    "total_assets",
    "total_equity",
    "total_debt",
    "current_assets",
    "current_liabilities",
    "interest_expense",
    "eps",
    "previous_eps",
}


def normalize_financial_data(
    raw_data: Mapping[str, float],
) -> FinancialInput:
    """
    Validate and normalize raw financial statement data.

    This function acts as the boundary between an external financial
    data source and the internal ratio-analysis system.

    Args:
        raw_data: Mapping containing normalized financial statement
            field names and their numeric values.

    Returns:
        Validated FinancialInput instance.

    Raises:
        ValueError: If required financial fields are missing.
    """
    missing_fields = REQUIRED_FINANCIAL_FIELDS - raw_data.keys()

    if missing_fields:
        missing = ", ".join(sorted(missing_fields))

        raise ValueError(
            f"Missing required financial fields: {missing}"
        )

    logger.info("Normalizing financial statement data.")

    return FinancialInput(
        revenue=raw_data["revenue"],
        previous_revenue=raw_data["previous_revenue"],
        ebitda=raw_data["ebitda"],
        ebit=raw_data["ebit"],
        net_profit=raw_data["net_profit"],
        total_assets=raw_data["total_assets"],
        total_equity=raw_data["total_equity"],
        total_debt=raw_data["total_debt"],
        current_assets=raw_data["current_assets"],
        current_liabilities=raw_data["current_liabilities"],
        interest_expense=raw_data["interest_expense"],
        eps=raw_data["eps"],
        previous_eps=raw_data["previous_eps"],
    )