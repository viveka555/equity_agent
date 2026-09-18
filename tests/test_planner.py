"""
Unit tests for planner.py
"""

from models.planner_models import classify_task, extract_company


def test_extract_company():
    assert extract_company("Analyze TCS") == "Tcs"


def test_classify_dcf():
    assert classify_task("Find intrinsic value") == "dcf"


def test_classify_news():
    assert classify_task("Latest news of BEL") == "news"