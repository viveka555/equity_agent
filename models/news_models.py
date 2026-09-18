"""
news_models.py

Structured output models for the News Agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

class NewsSummary(BaseModel):
    """
    Structured financial news summary.
    """
    company: str = Field(description="Comapny name")

    summary: str = Field(
        description="Short executive summary")

    key_events: list[str] = Field(
        description="Important news events")

    sentiment: str = Field(
        description="Bullish or Bearish, Neutral"
    )