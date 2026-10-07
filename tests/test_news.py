"""
test_news.py

Tests for the News Research Agent.

Responsibilities
----------------
- Verify that the News Agent can execute successfully.
- Verify that retrieved news is stored in graph state.
"""

from langchain_core.messages import HumanMessage

from agents.news import news_node


def test_news_node() -> None:
    """
    Verify that the News Agent retrieves company news
    and stores it in the graph state.

    Raises:
        AssertionError: If the news result is empty.
    """
    state = {
        "messages": [
            HumanMessage(content="What is the latest news about BEL?")
        ],
        "company": "BEL",
        "task": "news",
        "news": "",
        "final_report": "",
    }

    updated_state = news_node(state)

    assert updated_state["news"].strip(), (
        "News Agent returned an empty news result."
    )