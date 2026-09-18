"""
Unit tests for the News Agent.
"""

from agents.planner import classify_task


def test_news_routing():
    """
    Verify planner correctly routes
    news-related questions.
    """
    assert classify_task(
        "Latest news of BEL"
    ) == "news"