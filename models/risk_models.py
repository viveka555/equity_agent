"""
risk_models.py

Validated output models for evidence-based company risk analysis.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "moderate", "high", "critical"]
RiskCategory = Literal[
    "financial",
    "business",
    "market",
    "operational",
    "governance",
    "liquidity",
    "valuation",
    "other",
]
RiskSource = Literal[
    "financial_data",
    "ratio_analysis",
    "dcf_analysis",
    "news",
]


class RiskFinding(BaseModel):
    """Represent one company risk and the evidence supporting it."""

    category: RiskCategory = Field(
        description="The risk area most closely related to this finding."
    )
    severity: RiskLevel = Field(
        description="Qualitative severity supported by the evidence."
    )
    title: str = Field(description="Short name for the risk.")
    evidence: str = Field(
        description="Specific supplied fact supporting the finding."
    )
    source: RiskSource = Field(
        description="State input from which the evidence was taken."
    )
    implications: str = Field(
        description="Potential impact if the risk materializes."
    )
    mitigants: list[str] = Field(
        description="Evidence-backed factors that may reduce the risk."
    )


class RiskAnalysis(BaseModel):
    """Represent the overall risk assessment and evidence gaps."""

    overall_risk_level: RiskLevel = Field(
        description="Qualitative overall risk level."
    )
    summary: str = Field(
        description="Concise synthesis of the supplied risk evidence."
    )
    key_risks: list[RiskFinding] = Field(
        description="Material risks supported by supplied evidence."
    )
    data_gaps: list[str] = Field(
        description="Important risk information not present in the inputs."
    )
