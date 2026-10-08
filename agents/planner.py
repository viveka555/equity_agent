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

from config.llm import llm

from langchain_core.messages import HumanMessage

from state import GraphState

from models.planner_models import PlannerDecision

# ---------------------------------------------------------------------
# Planner System Prompt
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """
You are the Planner Agent for an Institutional Equity Research System.

Your only responsibility is to determine:

1. Company name, using recent conversation context when the latest
   message refers to a company indirectly
2. Next workflow task, based primarily on the latest user message

Available tasks:
- news   : Latest company news
- rag    : Annual report or PDF questions
- ratio  : Financial ratio analysis
- dcf    : Intrinsic value / DCF valuation
- risk   : Company risk analysis using supplied financial, valuation,
           and news evidence
- report : Complete institutional equity research report
- unknown: If the request does not match any workflow

Return only the structured output defined by the schema.
Do not answer the user's financial question.
"""

logger = logging.getLogger(__name__)



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
    logger.info("Running LLM Planner")

    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, HumanMessage):
        logger.warning("Last message was not Humanmessage")
        return state

    # Create a structured-output version of the LLM
    planner_llm = llm.with_structured_output(PlannerDecision)

    # Include bounded conversation history so follow-up questions can
    # retain their company context.
    decision = planner_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            *messages,
        ]
    )

    logger.info("Company detected: %s", decision.company)
    logger.info("Task detected: %s", decision.task)


    return {
        **state,
        "company": decision.company,
        "task" : decision.task
    }
