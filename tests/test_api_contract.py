"""Lightweight API contract checks that do not load the live Agent graph."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.main as api_main  # noqa: E402
from app.schemas import ChatRequest, ContractReviewRequest  # noqa: E402


def test_health_contract() -> None:
    response = asyncio.run(api_main.health())
    assert response.status == "ok"


def test_chat_contract_without_loading_rag() -> None:
    original_agent = api_main.run_agent_chat

    def fake_agent(query: str):
        assert query == "试用期最长多久？"
        return {
            "answer": "示例答案",
            "citations": ["示例引用"],
            "route": "law_qa",
            "intent": "law_qa",
            "tools_used": ["search_law_articles"],
            "result_type": "law_qa",
            "latency": 0.01,
        }

    api_main.run_agent_chat = fake_agent
    try:
        response = asyncio.run(api_main.chat(ChatRequest(query="  试用期最长多久？  ")))
    finally:
        api_main.run_agent_chat = original_agent

    assert response.answer == "示例答案"
    assert response.citations == ["示例引用"]
    assert response.route == "law_qa"
    assert response.intent == "law_qa"
    assert response.tools_used == ["search_law_articles"]
    assert response.result_type == "law_qa"
    assert response.latency == 0.01


def test_chat_contract_review_returns_structured_payload() -> None:
    original_agent = api_main.run_agent_chat

    def fake_agent(query: str):
        assert query == "请审查劳动合同。"
        return {
            "answer": "合同审查完成，整体风险等级为 medium。",
            "citations": ["法律依据"],
            "route": "contract_review",
            "intent": "contract_review",
            "tools_used": ["contract_review_rules"],
            "result_type": "contract_review",
            "risk_level": "medium",
            "review_status": "not_required",
            "review_id": None,
            "contract_review": {"risk_level": "medium", "findings": []},
            "latency": 0.01,
        }

    api_main.run_agent_chat = fake_agent
    try:
        response = asyncio.run(api_main.chat(ChatRequest(query="请审查劳动合同。")))
    finally:
        api_main.run_agent_chat = original_agent

    assert response.result_type == "contract_review"
    assert response.risk_level == "medium"
    assert response.review_status == "not_required"
    assert response.contract_review == {"risk_level": "medium", "findings": []}


def test_contract_review_contract_without_loading_rag() -> None:
    original_tool = api_main.review_labor_contract

    def fake_review_labor_contract(contract_text: str, include_evidence: bool = False):
        assert contract_text == "试用期一年。员工自愿放弃社保。"
        assert include_evidence is True
        return {
            "risk_level": "high",
            "review_status": "pending_review",
            "review_id": "review-1",
            "contract_review": {
                "risk_level": "high",
                "findings": [],
                "evidence": [],
                "suggestions": ["高风险合同审查结果已进入人工复核队列，审批通过后再返回完整审查结果。"],
                "disclaimer": "仅供参考，需人工复核",
                "latency": 0.01,
                "review_status": "pending_review",
                "review_id": "review-1",
            },
        }

    api_main.review_labor_contract = fake_review_labor_contract
    try:
        response = asyncio.run(
            api_main.review_contract(
                ContractReviewRequest(contract_text="  试用期一年。员工自愿放弃社保。  ")
            )
        )
    finally:
        api_main.review_labor_contract = original_tool

    assert response.risk_level == "high"
    assert response.findings == []
    assert response.evidence == []
    assert response.disclaimer == "仅供参考，需人工复核"
    assert response.latency == 0.01
    assert response.review_status == "pending_review"
    assert response.review_id == "review-1"


def test_contract_review_rejects_blank_text() -> None:
    try:
        ContractReviewRequest(contract_text="  ")
    except ValidationError:
        return
    raise AssertionError("blank contract_text should be rejected")


def test_review_api_contracts() -> None:
    original_review_service = api_main.review_service

    class FakeReviewService:
        def list_pending_reviews(self):
            return [
                {
                    "review_id": "review-1",
                    "source_type": "contract_review",
                    "status": "pending_review",
                    "payload": {"risk_level": "high"},
                    "created_at": "2026-05-13T00:00:00+00:00",
                    "updated_at": "2026-05-13T00:00:00+00:00",
                }
            ]

        def approve_review(self, review_id: str):
            assert review_id == "review-1"
            return {
                "review_id": review_id,
                "status": "approved",
                "final_answer": {"risk_level": "high", "disclaimer": "仅供参考，需人工复核"},
                "message": "审批通过，返回最终答案。",
            }

        def reject_review(self, review_id: str):
            assert review_id == "review-1"
            return {
                "review_id": review_id,
                "status": "rejected",
                "final_answer": None,
                "message": "审批拒绝，最终答案不予输出。",
            }

    api_main.review_service = FakeReviewService()
    try:
        pending = asyncio.run(api_main.list_pending_reviews())
        approved = asyncio.run(api_main.approve_review("review-1"))
        rejected = asyncio.run(api_main.reject_review("review-1"))
    finally:
        api_main.review_service = original_review_service

    assert pending.reviews[0].review_id == "review-1"
    assert pending.reviews[0].status == "pending_review"
    assert approved.status == "approved"
    assert approved.final_answer["risk_level"] == "high"
    assert rejected.status == "rejected"
    assert rejected.final_answer is None


if __name__ == "__main__":
    test_health_contract()
    test_chat_contract_without_loading_rag()
    test_chat_contract_review_returns_structured_payload()
    test_contract_review_contract_without_loading_rag()
    test_contract_review_rejects_blank_text()
    test_review_api_contracts()
    print("api contract ok")
