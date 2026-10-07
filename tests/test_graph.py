"""
test_graph.py

Tests for the LangGraph workflow.

Responsibilities
----------------
- Verify that the graph can be built successfully.
- Verify that planner decisions route to the correct nodes.
- Verify that the Ratio Agent is registered in the graph.
"""

from graph import build_graph, route_after_planner


def test_route_after_planner_ratio() -> None:
    """
    Verify that a ratio task is routed to the Ratio Agent.
    """

    state = {
        "task": "ratio",
    }

    assert route_after_planner(state) == "ratio"


def test_build_graph() -> None:
    """
    Verify that the complete LangGraph workflow compiles successfully.
    """

    graph = build_graph()

    assert graph is not None