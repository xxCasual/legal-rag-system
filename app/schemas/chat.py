"""API schemas for the legal RAG assistant."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned


class ChatResponse(BaseModel):
    answer: str
    citations: List[str]
    route: str
    intent: str
    intent_source: str = ""
    intent_confidence: float = Field(default=0.0, ge=0)
    tools_used: List[str]
    result_type: Literal["law_qa", "policy_qa", "contract_review", "refusal"] | str = ""
    risk_level: Literal["low", "medium", "high"] | None = None
    review_status: Literal["not_required", "pending_review"] | None = None
    review_id: str | None = None
    contract_review: Dict[str, Any] | None = None
    latency: float = Field(..., ge=0)


class ErrorResponse(BaseModel):
    error: str
    detail: str
