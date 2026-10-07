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
  ├── News → News Agent → Report → END
  ├── RAG  → RAG Agent → END
  └── Report → Report Agent → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.news import news_node
from agents.planner import planner_node
from agents.rag_agent import rag_node
from agents.report import report_node
from agents.ratio_agent import ratio_node
from state import GraphState


def route_after_planner(state: GraphState) -> str:
    return state["task"]


def build_graph():
    """
    Build and compile the LangGraph workflow.

    Returns:
        Compiled LangGraph application.
    """
    builder = StateGraph(GraphState)

    # Register graph nodes.
    builder.add_node("planner", planner_node)
    builder.add_node("news", news_node)
    builder.add_node("rag", rag_node)
    builder.add_node("report", report_node)
    builder.add_node("ratio", ratio_node)

    # Start workflow with the planner.
    builder.add_edge(START, "planner")

    # Route according to planner decision.
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "news": "news",
            "rag": "rag",
            "report": "report",
            "ratio":"ratio"
        },
    )

    # News continues through the report node.
    builder.add_edge("news", "report")

    # RAG already produces the answer.
    builder.add_edge("rag", END)

    # Report is the final node for report workflows.
    builder.add_edge("report", END)

    # Ratio analysis is the final node for ratio workflows.
    builder.add_edge("ratio", END)

    
    return builder.compile()