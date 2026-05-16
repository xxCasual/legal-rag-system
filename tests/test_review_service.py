"""Lightweight checks for JSON-backed human review service."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.review_service import (  # noqa: E402
    APPROVED,
    PENDING_REVIEW,
    REJECTED,
    ReviewNotFoundError,
    ReviewService,
    ReviewStateError,
)


def test_review_service_approves_pending_record() -> None:
    with TemporaryDirectory() as temp_dir:
        service = ReviewService(Path(temp_dir) / "pending_reviews.json")
        pending = service.create_pending_review(
            source_type="contract_review",
            payload={"risk_level": "high"},
            final_answer={"risk_level": "high", "answer": "完整审查结果"},
        )

        assert pending["status"] == PENDING_REVIEW
        assert "final_answer" not in pending
        assert service.list_pending_reviews()[0]["review_id"] == pending["review_id"]

        approved = service.approve_review(pending["review_id"])

        assert approved["status"] == APPROVED
        assert approved["final_answer"]["answer"] == "完整审查结果"
        assert service.list_pending_reviews() == []


def test_review_service_rejects_pending_record_without_final_answer() -> None:
    with TemporaryDirectory() as temp_dir:
        service = ReviewService(Path(temp_dir) / "pending_reviews.json")
        pending = service.create_pending_review(
            source_type="contract_review",
            payload={"risk_level": "high"},
            final_answer={"answer": "不应输出"},
        )

        rejected = service.reject_review(pending["review_id"])

        assert rejected["status"] == REJECTED
        assert rejected["final_answer"] is None
        assert service.list_pending_reviews() == []


def test_review_service_blocks_duplicate_decisions_and_missing_ids() -> None:
    with TemporaryDirectory() as temp_dir:
        service = ReviewService(Path(temp_dir) / "pending_reviews.json")
        pending = service.create_pending_review(
            source_type="contract_review",
            payload={"risk_level": "high"},
            final_answer={"answer": "完整审查结果"},
        )
        service.approve_review(pending["review_id"])

        try:
            service.reject_review(pending["review_id"])
        except ReviewStateError:
            pass
        else:
            raise AssertionError("decided review should not be decided again")

        try:
            service.approve_review("missing-review")
        except ReviewNotFoundError:
            pass
        else:
            raise AssertionError("missing review id should raise ReviewNotFoundError")


if __name__ == "__main__":
    test_review_service_approves_pending_record()
    test_review_service_rejects_pending_record_without_final_answer()
    test_review_service_blocks_duplicate_decisions_and_missing_ids()
    print("review service ok")
