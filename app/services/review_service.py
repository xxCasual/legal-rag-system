"""Local JSON-backed human review queue."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4

from app.core.config import settings


PENDING_REVIEW = "pending_review"
APPROVED = "approved"
REJECTED = "rejected"


class ReviewNotFoundError(KeyError):
    """Raised when a review id does not exist."""


class ReviewStateError(ValueError):
    """Raised when a review can no longer be approved or rejected."""


class ReviewService:
    """Persist pending human review records in a local JSON file."""

    def __init__(self, registry_path: Path | str | None = None) -> None:
        self.registry_path = Path(registry_path or settings.pending_reviews_path)
        self._lock = Lock()

    def create_pending_review(
        self,
        source_type: str,
        payload: Dict[str, Any],
        final_answer: Dict[str, Any],
    ) -> Dict[str, Any]:
        now = self._now()
        record = {
            "review_id": str(uuid4()),
            "source_type": source_type,
            "status": PENDING_REVIEW,
            "payload": payload,
            "final_answer": final_answer,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            records = self._load_records()
            records.append(record)
            self._write_records(records)
        return self._public_record(record)

    def list_pending_reviews(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                self._public_record(record)
                for record in self._load_records()
                if record.get("status") == PENDING_REVIEW
            ]

    def approve_review(self, review_id: str) -> Dict[str, Any]:
        return self._decide_review(
            review_id=review_id,
            status=APPROVED,
            message="审批通过，返回最终答案。",
            include_final_answer=True,
        )

    def reject_review(self, review_id: str) -> Dict[str, Any]:
        return self._decide_review(
            review_id=review_id,
            status=REJECTED,
            message="审批拒绝，最终答案不予输出。",
            include_final_answer=False,
        )

    def _decide_review(
        self,
        review_id: str,
        status: str,
        message: str,
        include_final_answer: bool,
    ) -> Dict[str, Any]:
        with self._lock:
            records = self._load_records()
            record = self._find_record(records, review_id)
            if record.get("status") != PENDING_REVIEW:
                raise ReviewStateError(f"Review {review_id} is already {record.get('status')}")
            record["status"] = status
            record["updated_at"] = self._now()
            self._write_records(records)

        return {
            "review_id": record["review_id"],
            "status": record["status"],
            "final_answer": record.get("final_answer") if include_final_answer else None,
            "message": message,
        }

    def _find_record(self, records: List[Dict[str, Any]], review_id: str) -> Dict[str, Any]:
        for record in records:
            if record.get("review_id") == review_id:
                return record
        raise ReviewNotFoundError(review_id)

    def _public_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "review_id": record["review_id"],
            "source_type": record["source_type"],
            "status": record["status"],
            "payload": record.get("payload", {}),
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
        }

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()


review_service = ReviewService()
