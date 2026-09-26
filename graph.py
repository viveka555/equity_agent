"""
graph.py

LangGraph workflow definition for the Institutional Equity Research Agent.

Responsibilities
----------------
- Register graph nodes
- Define workflow edges
- Perform conditional routing
- Compile and return the graph

Current Workflow
----------------
START
  ↓
Planner
  ↓
News / Report
  ↓
END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from state import GraphState
from agents.planner import planner_node
from agents.news import news_node
from agents.report import report_node


def route_after_planner(state: GraphState) -> str:
    """
    Decide which node should execute after the planner.

    Args:
        state: Shared LangGraph state.

    Returns:
        Name of the next node.
    """
    if state["task"] == "news":
        return "news"

    # For now every other task goes to report.
    # Later we will add RAG, Ratio and DCF.
    return "report"


def build_graph():
    """
    Build and compile the LangGraph workflow.

    Returns:
        Compiled LangGraph application.
    """
    builder = StateGraph(GraphState)

    # -------------------------
    # Register Nodes
    # -------------------------
    builder.add_node("planner", planner_node)
    builder.add_node("news", news_node)
    builder.add_node("report", report_node)

    # -------------------------
    # Workflow
    # -------------------------
    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "news": "news",
            "report": "report",
        },
    )

    builder.add_edge("news", "report")
    builder.add_edge("report", END)

    return builder.compile()