"""
report.py

Placeholder report node.

Future phases will generate institutional investment reports.
"""
from __future__ import annotations

from state import GraphState

def report_node(state:GraphState)-> GraphState:
    """
    Create a temporary response.

    Args:
        state: Current graph state.

    Returns:
        Updated graph state.
    """
    task = state["task"]
    company = state["company"]

    response = (
        f"Planner routed company={company}, task={task}.",
        "Specialist agents will be added in future phases.")

    return {
        **state,
        "Final_report":response
    }
