"""
report.py

Evidence-grounded institutional research report generation node.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config.llm import llm
from langchain_core.messages import HumanMessage

from models.report_models import InstitutionalResearchReport
from state import GraphState

logger = logging.getLogger(__name__)

REPORT_INPUT_KEYS = (
    "news",
    "annual_report_analysis",
    "financial_data",
    "ratio_analysis",
    "dcf_analysis",
    "risk_analysis",
)

SYSTEM_PROMPT = """
You are an institutional equity research report editor. Synthesize only the
evidence included in the request. Do not use outside knowledge, invent facts,
metrics, citations, company details, catalysts, risks, or mitigants. Identify
which source key supports each factual claim, and include only evidence keys
present in the request. Separate facts from interpretation. If evidence is
missing or incomplete, say so in data_gaps and the relevant section. Never
create a buy/sell recommendation, price target, or unsupported valuation view.
Do not treat missing evidence as proof of a negative fact. Return only the
structured output requested by the schema.
""".strip()

_MISSING_INPUT_DESCRIPTIONS = {
    "news": "No company news evidence was supplied.",
    "annual_report_analysis": "No annual-report research answer was supplied.",
    "financial_data": "No normalized financial statement data was supplied.",
    "ratio_analysis": "No financial ratio analysis was supplied.",
    "dcf_analysis": "No DCF valuation analysis was supplied.",
    "risk_analysis": "No structured risk analysis was supplied.",
}


def _available_evidence(state: GraphState) -> dict[str, Any]:
    """Collect populated research outputs from shared graph state.

    Args:
        state: Shared graph state.

    Returns:
        Mapping containing only non-empty report evidence sources.
    """
    return {
        key: state[key]
        for key in REPORT_INPUT_KEYS
        if state.get(key) not in (None, "", {}, [])
    }


def _user_question(state: GraphState) -> str:
    """Return the latest human message, if one is available.

    Args:
        state: Shared graph state.

    Returns:
        Latest human message content or an empty string.
    """
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1], HumanMessage):
        content = messages[-1].content
        return content if isinstance(content, str) else str(content)
    return ""


def _report_without_evidence(company: str) -> InstitutionalResearchReport:
    """Build a transparent report shell when no research data is present.

    Args:
        company: Company name from graph state.

    Returns:
        Validated report that identifies unavailable inputs.
    """
    gaps = list(_MISSING_INPUT_DESCRIPTIONS.values())
    return InstitutionalResearchReport(
        executive_summary=(
            f"A research report for {company} cannot be completed because "
            "no company-specific research evidence was supplied."
        ),
        business_and_news="Not assessed: no company news or annual-report evidence was supplied.",
        financial_analysis="Not assessed: no financial statements or ratio analysis were supplied.",
        valuation_analysis="Not assessed: no DCF valuation was supplied.",
        risk_analysis="Not assessed: no structured risk analysis was supplied.",
        catalysts=[],
        overall_view="insufficient_evidence",
        conclusion=(
            "Provide company-specific research inputs before drawing an "
            "investment conclusion. Missing data is not evidence of poor "
            "company performance."
        ),
        evidence=[],
        data_gaps=gaps,
    )


def _format_report(
    company: str,
    report: InstitutionalResearchReport,
) -> str:
    """Render validated report sections as a readable plain-text report.

    Args:
        company: Company name from graph state.
        report: Validated structured research report.

    Returns:
        User-facing report text.
    """
    sections = [
        f"Institutional Equity Research Report — {company}",
        "",
        "Executive summary",
        report.executive_summary,
        "",
        "Business and news",
        report.business_and_news,
        "",
        "Financial analysis",
        report.financial_analysis,
        "",
        "Valuation analysis",
        report.valuation_analysis,
        "",
        "Risk analysis",
        report.risk_analysis,
        "",
        "Overall view",
        f"{report.overall_view.replace('_', ' ').title()}: {report.conclusion}",
    ]

    if report.catalysts:
        sections.extend(["", "Potential catalysts"])
        sections.extend(f"- {catalyst}" for catalyst in report.catalysts)

    if report.evidence:
        sections.extend(["", "Evidence"])
        sections.extend(
            f"- {item.claim} [source: {item.source}; evidence: "
            f"{item.supporting_evidence}]"
            for item in report.evidence
        )

    if report.data_gaps:
        sections.extend(["", "Data gaps and limitations"])
        sections.extend(f"- {gap}" for gap in report.data_gaps)

    sections.extend(
        [
            "",
            "This report summarizes supplied evidence and is not a "
            "personalized investment recommendation.",
        ]
    )
    return "\n".join(sections)


def _generate_institutional_report(
    state: GraphState,
    company: str,
    evidence: dict[str, Any],
) -> InstitutionalResearchReport:
    """Generate and validate a report from supplied state evidence.

    Args:
        state: Shared graph state, used to read the user's request.
        company: Company name from graph state.
        evidence: Non-empty supported evidence sources.

    Returns:
        Validated structured research report.

    Raises:
        ValueError: If the model cites an evidence source not supplied.
    """
    request = {
        "company": company,
        "question": _user_question(state),
        "evidence": evidence,
    }
    structured_llm = llm.with_structured_output(InstitutionalResearchReport)
    result = structured_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", json.dumps(request, indent=2, default=str)),
        ]
    )
    if not isinstance(result, InstitutionalResearchReport):
        result = InstitutionalResearchReport.model_validate(result)

    unsupported_sources = {
        item.source for item in result.evidence
    } - evidence.keys()
    if unsupported_sources:
        raise ValueError(
            "Report cited evidence sources that were not supplied: "
            + ", ".join(sorted(unsupported_sources))
        )
    return result


def report_node(state: GraphState) -> GraphState:
    """Generate a report or render the existing news workflow output.

    For ``task='report'``, the node synthesizes supplied research evidence.
    It returns a gap report without calling the LLM when no evidence exists.
    For the existing news route, it preserves the deterministic news report.

    Args:
        state: Shared LangGraph state.

    Returns:
        Updated graph state containing ``final_report`` and, for full
        research requests, a serializable ``institutional_report``.
    """
    company = state.get("company", "unknown")
    task = state.get("task", "")

    if task == "report":
        evidence = _available_evidence(state)
        logger.info(
            "Generating institutional report for %s from sources: %s",
            company,
            ", ".join(evidence) if evidence else "none",
        )
        if evidence:
            report = _generate_institutional_report(
                state=state,
                company=company,
                evidence=evidence,
            )
        else:
            report = _report_without_evidence(company)

        return {
            **state,
            "institutional_report": report.model_dump(),
            "final_report": _format_report(company, report),
        }

    if task == "news":
        final_report = (
            f"Latest News Report\n\nCompany: {company}\n\n"
            f"{state.get('news', '')}"
        )
    else:
        final_report = (
            f"Planner selected task: {task}\nCompany: {company}"
        )

    return {**state, "final_report": final_report}
