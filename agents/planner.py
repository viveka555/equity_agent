"""
planner.py

Planner Agent for the Institutional Equity Research System.

Responsibilities
----------------
- Understand user intent.
- Extract company name.
- Decide which specialist agent should execute.

Used by
--------
graph.py
"""
from __future__ import annotations
import logging
import re

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph

from state import GraphState
from config.settings import(
    TASK_DCF,
    TASK_NEWS,
    TASK_RAG,
    TASK_RATIO,
    TASK_REPORT,
    TASK_UNKNOWN
)

logger = logging.getLogger(__name__)

COMPANIES = [
    "tcs",
    "infosys",
    "bel",
    "yes bank",
    "ltf",
    "eternal",
    "max healthcare",
]

def extract_company(text:str) ->str:
    """
    Extract a known company name from user input.

    Args:
        text: User message.

    Returns:
        Company name if found, otherwise empty string.
    """
    text = text.lower()

    for company in COMPANIES:
        if company in text:
            return company.title()

    return ""

def classify_task(text: str) -> str:
    """
    Classify the user's intent into a planner task.

    Args:
        text: User message.

    Returns:
        One of the predefined task names.
    """
    text = text.lower().strip()

    # 1. DCF / Valuation (highest priority)
    if any(keyword in text for keyword in [
        "dcf",
        "intrinsic value",
        "fair value",
        "valuation",
        "discounted cash flow",
    ]):
        return TASK_DCF

    # 2. News
    if any(keyword in text for keyword in [
        "news",
        "latest",
        "headline",
        "announcement",
    ]):
        return TASK_NEWS

    # 3. Annual report / RAG
    if any(keyword in text for keyword in [
        "annual report",
        "10-k",
        "10k",
        "pdf",
        "financial statement",
    ]):
        return TASK_RAG

    # 4. Financial ratios
    if any(keyword in text for keyword in [
        "ratio",
        "roe",
        "roce",
        "eps",
        "debt",
        "margin",
    ]):
        return TASK_RATIO

    # 5. Full institutional analysis
    if any(keyword in text for keyword in [
        "analyze",
        "analysis",
        "full report",
    ]):
        return TASK_REPORT

    return TASK_UNKNOWN

def planner_node(state: GraphState) ->GraphState:
    """
    LangGraph planner node.

    Reads the latest user message and updates the graph state
    with the detected company and task.

    Args:
        state: Current graph state.

    Returns:
        Updated graph state.
    """
    logger.info("Planner node started")

    last_messge = state["messages"][-1]  

    if not isinstance(last_messge, HumanMessage):
        logger.warning("Last message was not Humanmessage")
        return state

    company = extract_company(last_messge.content)
    task = classify_task(last_messge.content)

    logger.info("Company detected %s", company)
    logger.info("Task dected %s", task)

    return {
        **state,
        "Company": company,
        "Task" : task
    }