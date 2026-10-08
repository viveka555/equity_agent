"""
risk_agent.py

Evidence-based Risk Analysis Agent for the Institutional Equity
Research System.

The agent synthesizes supplied financial data, ratio analysis, DCF
analysis, and company news. It does not fetch data or infer unsupported
company facts.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config.llm import llm
from langchain_core.messages import HumanMessage

from models.risk_models import RiskAnalysis
from state import GraphState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are the Risk Analysis Agent for an institutional equity research system.

Assess the company using only the evidence supplied in the user message.
Do not invent facts, metrics, sources, or mitigants. Cite each finding with
one of the supplied source keys (financial_data, ratio_analysis,
dcf_analysis, or news) and quote or accurately describe its supporting
evidence. Identify material missing information in data_gaps. If evidence
is limited, state that limitation in the summary and choose a cautious
qualitative risk level. Do not treat a risk as established solely because
data is absent. Return only the structured output requested by the schema.
""".strip()

RISK_INPUT_KEYS = (
    "financial_data",
    "ratio_analysis",
    "dcf_analysis",
    "news",
)


def _available_evidence(state: GraphState) -> dict[str, Any]:
    """Collect non-empty evidence sources from graph state.

    Args:
        state: Shared graph state.

    Returns:
        Mapping of evidence-source names to their supplied values.
    """
    return {
        key: state[key]
        for key in RISK_INPUT_KEYS
        if state.get(key) not in (None, "", {})
    }


def _format_report(company: str, analysis: RiskAnalysis) -> str:
    """Create a readable risk summary from validated model output.

    Args:
        company: Company name in graph state.
        analysis: Validated risk analysis.

    Returns:
        Human-readable summary for ``final_report``.
    """
    lines = [
        f"Risk Analysis — {company}",
        "",
        f"Overall risk level: {analysis.overall_risk_level.title()}",
        analysis.summary,
    ]

    if analysis.key_risks:
        lines.extend(["", "Key risks:"])
        for finding in analysis.key_risks:
            lines.append(
                f"- [{finding.severity.title()}] {finding.title} "
                f"({finding.category}; source: {finding.source}): "
                f"{finding.evidence}"
            )
            lines.append(f"  Potential impact: {finding.implications}")
            if finding.mitigants:
                lines.append(
                    "  Evidence-backed mitigants: "
                    + "; ".join(finding.mitigants)
                )

    if analysis.data_gaps:
        lines.extend(["", "Information gaps:"])
        lines.extend(f"- {gap}" for gap in analysis.data_gaps)

    return "\n".join(lines)


def risk_node(state: GraphState) -> GraphState:
    """Analyze company risks from evidence already present in graph state.

    Args:
        state: Shared graph state. At least one non-empty evidence source
            among financial_data, ratio_analysis, dcf_analysis, or news is
            required.

    Returns:
        Updated graph state with structured ``risk_analysis`` and a
        readable ``final_report``.

    Raises:
        ValueError: If none of the supported evidence sources is supplied.
    """
    company = state.get("company", "unknown")
    evidence = _available_evidence(state)
    if not evidence:
        raise ValueError(
            f"No risk-analysis evidence is available for company: {company}"
        )

    question = ""
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], HumanMessage):
        question = str(messages[-1].content)

    request = {
        "company": company,
        "question": question,
        "evidence": evidence,
    }
    logger.info(
        "Running Risk Analysis Agent for %s using sources: %s",
        company,
        ", ".join(evidence),
    )

    structured_llm = llm.with_structured_output(RiskAnalysis)
    analysis = structured_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", json.dumps(request, indent=2, default=str)),
        ]
    )
    if not isinstance(analysis, RiskAnalysis):
        analysis = RiskAnalysis.model_validate(analysis)

    logger.info("Risk analysis completed for %s", company)
    return {
        **state,
        "risk_analysis": analysis.model_dump(),
        "final_report": _format_report(company, analysis),
    }
