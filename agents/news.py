"""
news.py

News Research Agent.

Responsibilities
----------------
- Fetch recent company news.
- Store the retrieved news in the shared LangGraph state.

Used by
--------
graph.py
"""

from __future__ import annotations

import logging

from state import GraphState
from tools.web_tools import search_company_news

logger = logging.getLogger(__name__)


def news_node(state: GraphState) -> GraphState:
    """
    Execute the News Research Agent.

    Args:
        state: Shared LangGraph state containing the company name.

    Returns:
        Updated graph state containing the retrieved news.

    Raises:
        KeyError: If the company is missing from the graph state.
    """
    company = state["company"]

    logger.info("Running News Agent for %s", company)

    news_text = search_company_news.invoke(company)

    return {
        **state,
        "news": news_text,
    }