"""
planner_models.py

Pydantic models used by the LLM Planner Agent.

Responsibilities
----------------
- Define the structured output expected from the planner LLM.
- Validate company name and workflow task.
- Ensure LangGraph receives reliable routing decisions.

Used by
--------
agents/planner.py
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class PlannerDecision(BaseModel):
    """
    Structured routing decision returned by the Planner LLM.

    Attributes
    ----------
    company:
        Company identified from the user's query.

    task:
        Next specialist workflow to execute.
    """
    company : str = Field(
        ..., description="Company name such as BEL, TCS, Infosys, etc.",
        example=["BEL"],
    )
    task: Literal[
        "news",
        "rag",
        "ratio",
        "dcf",
        "risk",
        "report",
        "unknown"
    ] =Field(
        ..., description="Workflow task selected by the planner.",
        examples=["report"])
