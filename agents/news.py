"""
news.py

News Research Agent.

Responsibilities
----------------
- Fetch recent company news
- Summarize events
- Store structured results in graph state

Used by
--------
graph.py
"""
from __future__ import annotations

import logging

from tools.web_tools import search_company_news
from state import GraphState

logger = logging.getLogger(__name__)

def news_node(state:GraphState) ->GraphState:
    """
    Execute the News Research Agent.

    Args:
        state: Shared LangGraph state.

    Returns:
        Updated state containing news summary.
    """
    company = state["company"]

    logger.info("Running News Agent for %s", company)

    news_text = search_company_news.invoke(company)

    
    return {
    **state,
    "news": news_text,
}