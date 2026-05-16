"""API schemas for human review workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel


ReviewStatus = Literal["pending_review", "approved", "rejected"]


class PendingReviewRecord(BaseModel):
    review_id: str
    source_type: str
    status: ReviewStatus
    payload: Dict[str, Any]
    created_at: str
    updated_at: str


class PendingReviewListResponse(BaseModel):
    reviews: List[PendingReviewRecord]


class ReviewDecisionResponse(BaseModel):
    review_id: str
    status: ReviewStatus
    final_answer: Dict[str, Any] | None = None
    message: str
