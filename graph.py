"""
graph.py

LangGraph workflow definition.

Current flow

START
  ↓
Planner
  ↓
Report
  ↓
END
"""
from langgraph.graph import START, END, StateGraph

from state import GraphState

from agents.planner import planner_node
from agents.report import report_node

def build_graph():
    """
    Build and compile the LangGraph workflow.

    Returns:
        Compiled graph.
    """
    builder = StateGraph(GraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "report")
    builder.add_edge("report", END)

    return builder.compile()