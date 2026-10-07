"""
test_news.py

Tests for the News Research Agent.

The external web-search tool is mocked so the test remains
independent of network availability.
"""

from unittest.mock import patch

from langchain_core.messages import HumanMessage

from agents.news import news_node


def test_news_node() -> None:
    """
    Verify that the News Agent stores retrieved news in graph state.
    """

    state = {
        "messages": [
            HumanMessage(
                content="What is the latest news about BEL?"
            )
        ],
        "company": "BEL",
        "task": "news",
        "news": "",
        "final_report": "",
    }

    with patch(
        "agents.news.search_company_news",
        autospec=True,
    ) as mock_search:

        mock_search.invoke.return_value = (
            "BEL reported strong quarterly results."
        )

        updated_state = news_node(state)

    mock_search.invoke.assert_called_once_with("BEL")

    assert updated_state["news"] == (
        "BEL reported strong quarterly results."
    )