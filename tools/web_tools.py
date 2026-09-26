"""
web_tools.py

Web search tools used by the Institutional Equity Research Agent.

Responsibilities
----------------
- Search the web for recent financial news.
- Return clean text results.
- Handle search failures gracefully.

Used by
--------
agents/news.py
LangGraph ToolNode

Example
-------
search_company_news.invoke("BEL latest news")
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)

# Reuse one search client instead of creating it repeatedly.
search = DuckDuckGoSearchRun()

@tool
def search_company_news(company:str) ->str:
    """
    Search recent news about a company.
    Args:
        company: Company name.
    Returns:
        Concatenated news results.
    Raises:
        RuntimeError:
            If web search fails.
    """ 
    try:
        query = f"{company} latest financial news"

        logger.info("Searching news for %s", company)

        results = search.run(query)

        return results

    except Exception as exc:

        logger.exception("Web search failed!")
        
        raise RuntimeError("Unable to fetch company news.") from exc