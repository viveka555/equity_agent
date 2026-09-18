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

from models.planner_models import planner_node
from agents.report import report_node
from agents.news import news_node

def route_after_planner(state:GraphState) ->str:
    """
    Decide the next node after planner
    """
    if state["task"] == "news":
        return "news"

    return "report"

builder = StateGraph(GraphState)

builder.add_node("planner", planner_node)
builder.add_node("news", news_node)
builder.add_node("report", report_node)

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

graph = builder.compile()