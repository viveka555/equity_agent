"""
web_tools.py

Web search tools used by the Institutional Equity Research Agent.

Responsibilities
----------------
- Retrieve recent, structured financial news results.
- Present headlines with their source, date, summary, and link.
- Handle search failures with a clear error.

Used by
--------
agents/news.py
LangGraph ToolNode

Example
-------
search_company_news.invoke("BSE India")
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Request news articles for the Indian English-language market and preserve
# the metadata that lets users inspect each result themselves.
search = DuckDuckGoSearchAPIWrapper(
    region="in-en",
    max_results=5,
    source="news",
    time="m",
)


@tool
def search_company_news(company: str) -> str:
    """Search for recent financial news and format individual article results.

    Args:
        company: Company or market name to search for.

    Returns:
        A readable list of article headlines with dates, sources, snippets,
        and direct links. Returns an explicit message when no articles match.

    Raises:
        RuntimeError: If the web search service cannot be reached.
    """
    normalized_company = company.strip()
    if not normalized_company:
        return "No company or market name was provided for the news search."

    query = f"{normalized_company} latest financial market news"
    logger.info("Searching news for %s", normalized_company)

    try:
        results = search.results(
            query,
            max_results=search.max_results,
            source="news",
        )
    except Exception as exc:
        logger.exception("News search failed for %s", normalized_company)
        raise RuntimeError("Unable to fetch company news.") from exc

    formatted_results = _format_news_results(results, normalized_company)
    if not formatted_results:
        return f"No recent news articles were found for {normalized_company}."

    return (
        f"Recent news for {normalized_company} "
        f"({len(formatted_results)} articles):\n\n"
        + "\n\n".join(
            f"{index}. {article}"
            for index, article in enumerate(formatted_results, start=1)
        )
    )


def _format_news_results(
    results: list[dict[str, str]],
    company: str,
) -> list[str]:
    """Filter and format matching results as source-linked articles.

    Args:
        results: Article dictionaries returned by the news search backend.
        company: Company or market name the user asked about.

    Returns:
        Relevant formatted articles, skipping unrelated or untitled results.
    """
    company_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", company.casefold())
        if term
        not in {
            "a", "an", "and", "company", "corp", "corporation", "co",
            "inc", "india", "limited", "ltd", "of", "plc", "the",
        }
    }
    articles: list[str] = []
    for result in results:
        title = result.get("title", "").strip()
        if not title:
            continue

        searchable_text = " ".join(
            (title, result.get("snippet", ""))
        ).casefold()
        article_terms = set(re.findall(r"[a-z0-9]+", searchable_text))
        minimum_matches = min(2, len(company_terms))
        if company_terms and len(company_terms & article_terms) < minimum_matches:
            logger.info("Skipping search result unrelated to %s: %s", company, title)
            continue

        source = result.get("source", "").strip()
        link = result.get("link", "").strip()
        published = result.get("date", "").strip()
        snippet = result.get("snippet", "").strip()

        if not source and link:
            source = urlparse(link).netloc.removeprefix("www.")

        details: list[str] = []
        if source:
            details.append(f"Source: {source}")
        if published:
            details.append(f"Published: {published}")

        lines = [title]
        if details:
            lines.append("   " + " | ".join(details))
        if snippet:
            lines.append(f"   Summary: {snippet}")
        if link:
            lines.append(f"   Link: {link}")
        articles.append("\n".join(lines))

    return articles
