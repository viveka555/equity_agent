"""Tests for structured, company-relevant news search output."""

from unittest.mock import patch

import pytest
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from tools.web_tools import _format_news_results, search_company_news


def test_format_news_results_filters_unrelated_articles() -> None:
    """Keep matching BEL stories and omit articles about other companies."""
    results = [
        {
            "title": "Bharat Electronics Limited wins a defence contract",
            "source": "Example News",
            "date": "2026-10-01",
            "snippet": "Bharat Electronics received a new order.",
            "link": "https://example.com/bel-story",
        },
        {
            "title": "Bharat Dynamics signs a new contract",
            "source": "Example News",
            "date": "2026-10-01",
            "snippet": "A defence company received an order.",
            "link": "https://example.com/bdl-story",
        },
    ]

    articles = _format_news_results(results, "Bharat Electronics Limited")

    assert len(articles) == 1
    assert "Bharat Electronics Limited wins" in articles[0]
    assert "Source: Example News" in articles[0]
    assert "Published: 2026-10-01" in articles[0]
    assert "https://example.com/bel-story" in articles[0]


def test_news_search_reports_when_no_relevant_articles_exist() -> None:
    """Do not present unrelated search snippets as company news."""
    unrelated_result = {
        "title": "Bharat Dynamics signs a new contract",
        "source": "Example News",
        "date": "2026-10-01",
        "snippet": "A defence company received an order.",
        "link": "https://example.com/bdl-story",
    }

    with patch.object(
        DuckDuckGoSearchAPIWrapper,
        "results",
        return_value=[unrelated_result],
    ):
        result = search_company_news.invoke("Bharat Electronics Limited")

    assert "No recent news articles were found" in result


def test_news_search_wraps_search_service_failure() -> None:
    """Expose a clear tool error when the news provider is unavailable."""
    with patch.object(
        DuckDuckGoSearchAPIWrapper,
        "results",
        side_effect=OSError("offline"),
    ):
        with pytest.raises(RuntimeError, match="Unable to fetch company news"):
            search_company_news.invoke("Bharat Electronics Limited")
