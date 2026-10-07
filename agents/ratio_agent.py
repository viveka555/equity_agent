"""
ratio_agent.py

Financial Ratio Agent for the Institutional Equity Research System.

Responsibilities
----------------
- Receive the current LangGraph state.
- Obtain normalized financial data.
- Calculate financial ratios.
- Store the ratio analysis in graph state.

Architecture Role
-----------------
Planner
    ↓
Ratio Agent
    ↓
Financial Data Tool
    ↓
Ratio Calculator
    ↓
RatioAnalysis
    ↓
Graph State

Used by
--------
graph.py
"""

from __future__ import annotations

import logging

from state import GraphState
from tools.financial_tools import normalize_financial_data
from tools.ratio_calculator import calculate_ratios

logger = logging.getLogger(__name__)


def ratio_node(state: GraphState) -> GraphState:
    """
    Execute the Financial Ratio Agent.

    The current implementation uses normalized financial data
    supplied by the financial-data layer. Financial calculations
    are delegated to the ratio calculation engine.

    Args:
        state: Shared LangGraph state.

    Returns:
        Updated graph state containing the financial ratio analysis.

    Raises:
        ValueError: If required financial data is unavailable.
    """
    company = state["company"]

    logger.info("Running Ratio Agent for %s", company)

    # Temporary normalized financial-data boundary.
    # The external financial-data provider will be connected here
    # in the next integration step.
    raw_financial_data = state.get("financial_data")

    if not raw_financial_data:
        raise ValueError(
            f"No financial data available for company: {company}"
        )

    financial_data = normalize_financial_data(raw_financial_data)

    ratios = calculate_ratios(financial_data)

    logger.info(
        "Financial ratio analysis completed for %s",
        company,
    )

    return {
        **state,
        "ratio_analysis": ratios.model_dump(),
    }