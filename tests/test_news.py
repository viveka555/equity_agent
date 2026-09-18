"""
Unit tests for the News Agent.
"""

from models.planner_models import classify_task


def test_news_routing():
    """
    Verify planner correctly routes
    news-related questions.
    """
    assert classify_task(
        "Latest news of BEL"
    ) == "news"