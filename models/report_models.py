"""
report_models.py

Validated output models for the Institutional Research Report Generator.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ReportSource = Literal[
    "news",
    "annual_report_analysis",
    "financial_data",
    "ratio_analysis",
    "dcf_analysis",
    "risk_analysis",
]
ReportView = Literal[
    "positive",
    "balanced",
    "cautious",
    "insufficient_evidence",
]


class ReportEvidence(BaseModel):
    """Represent a report claim with a traceable graph-state source."""

    claim: str = Field(description="Concise factual claim in the report.")
    source: ReportSource = Field(
        description="Exact evidence source key supplied to the report agent."
    )
    supporting_evidence: str = Field(
        description="Specific fact or excerpt supporting the claim."
    )


class InstitutionalResearchReport(BaseModel):
    """Structured institutional-style synthesis of available research."""

    executive_summary: str = Field(
        description="Concise summary of supported findings and limitations."
    )
    business_and_news: str = Field(
        description="Business context and material supplied news evidence."
    )
    financial_analysis: str = Field(
        description="Financial performance and ratio analysis from supplied data."
    )
    valuation_analysis: str = Field(
        description="DCF valuation and assumptions from supplied analysis."
    )
    risk_analysis: str = Field(
        description="Evidence-backed risks and mitigants from supplied analysis."
    )
    catalysts: list[str] = Field(
        description="Potential catalysts explicitly supported by supplied evidence."
    )
    overall_view: ReportView = Field(
        description="Qualitative synthesis; never a buy, sell, or price target."
    )
    conclusion: str = Field(
        description="Decision-useful conclusion with appropriate uncertainty."
    )
    evidence: list[ReportEvidence] = Field(
        description="Key factual claims with source keys and supporting evidence."
    )
    data_gaps: list[str] = Field(
        description="Material missing inputs or limitations in the supplied evidence."
    )
