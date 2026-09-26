"""
report.py

Report generation node.

This node converts structured graph state into a user-facing report.
"""

from __future__ import annotations

from state import GraphState


def report_node(state: GraphState) -> GraphState:
    """
    Generate the final response.

    Args:
        state: Shared LangGraph state.

    Returns:
        Updated state containing final_report.
    """
    company = state["company"]
    task = state["task"]
    news = state.get("news", "")

    if task == "news":
        report = f"""Latest News Report

Company: {company}

{news}
"""
    else:
        report = (
            f"Planner selected task: {task}\n"
            f"Company: {company}"
        )

    return {
        **state,
        "final_report": report,
    }
