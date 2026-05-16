"""API schemas for labor contract risk review."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


RiskLevel = Literal["low", "medium", "high"]
ClauseStatus = Literal["present", "missing", "unclear"]
ClauseType = Literal[
    "probation",
    "term",
    "salary",
    "working_hours",
    "social_insurance",
    "termination",
    "non_compete",
]


class ContractReviewRequest(BaseModel):
    contract_text: str = Field(..., min_length=1)

    @field_validator("contract_text")
    @classmethod
    def contract_text_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("contract_text must not be blank")
        return cleaned


class ContractFinding(BaseModel):
    clause_type: ClauseType
    clause_name: str
    status: ClauseStatus
    risk_level: RiskLevel
    extracted_text: str
    analysis: str
    evidence: List[str]
    suggestion: str


class ContractReviewResponse(BaseModel):
    risk_level: RiskLevel
    findings: List[ContractFinding]
    evidence: List[str]
    suggestions: List[str]
    disclaimer: str
    latency: float = Field(..., ge=0)
    review_status: Literal["not_required", "pending_review"] = "not_required"
    review_id: str | None = None
