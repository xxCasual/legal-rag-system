"""Tests for the local Agent intent classifier."""

from __future__ import annotations

import json
from pathlib import Path

from app.agent.intent_classifier import IntentClassifier


ROOT = Path(__file__).resolve().parent.parent


def _expected_route(sample: dict[str, object]) -> str:
    expected_tools = set(sample.get("expected_tools", []))
    if "contract_review_rules" in expected_tools:
        return "contract_review"
    if sample.get("should_refuse"):
        return "refusal"
    return str(sample["expected_intent"])


def test_local_rules_cover_split_smoke_samples_without_remote_llm() -> None:
    classifier = IntentClassifier(enable_embedding_fallback=False)
    samples = json.loads((ROOT / "data" / "eval" / "test.json").read_text())

    mismatches = [
        (sample["query"], _expected_route(sample), classifier.classify(sample["query"]).intent)
        for sample in samples
        if classifier.classify(sample["query"]).intent != _expected_route(sample)
    ]

    assert mismatches == []


class _FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts, normalize_embeddings: bool = True):  # noqa: ANN001
        if isinstance(texts, str):
            return self.vectors[texts]
        return [self.vectors[text] for text in texts]


def test_embedding_fallback_handles_queries_not_caught_by_rules() -> None:
    query = "周末额外出勤的报酬应该如何处理？"
    classifier = IntentClassifier(
        embedder=_FakeEmbedder(
            {
                query: [0.96, 0.04, 0.0, 0.0],
                "法律规定、赔偿、工资、加班、仲裁、劳动合同法问题": [1.0, 0.0, 0.0, 0.0],
                "公司制度、员工手册、报销、考勤、请假、内部流程问题": [0.0, 1.0, 0.0, 0.0],
                "审查劳动合同文本、识别合同条款风险": [0.0, 0.0, 1.0, 0.0],
                "无关问题、违法违规请求、天气股票编程闲聊": [0.0, 0.0, 0.0, 1.0],
            }
        ),
        enable_embedding_fallback=True,
        embedding_threshold=0.5,
    )

    result = classifier.classify(query)

    assert result.intent == "law_qa"
    assert result.source == "embedding"
    assert result.confidence > 0.9


def test_low_confidence_embedding_falls_back_to_refusal() -> None:
    query = "帮我看看这个事情怎么办"
    classifier = IntentClassifier(
        embedder=_FakeEmbedder(
            {
                query: [0.25, 0.25, 0.25, 0.25],
                "法律规定、赔偿、工资、加班、仲裁、劳动合同法问题": [1.0, 0.0, 0.0, 0.0],
                "公司制度、员工手册、报销、考勤、请假、内部流程问题": [0.0, 1.0, 0.0, 0.0],
                "审查劳动合同文本、识别合同条款风险": [0.0, 0.0, 1.0, 0.0],
                "无关问题、违法违规请求、天气股票编程闲聊": [0.0, 0.0, 0.0, 1.0],
            }
        ),
        enable_embedding_fallback=True,
        embedding_threshold=0.8,
    )

    result = classifier.classify(query)

    assert result.intent == "refusal"
    assert result.source == "low_confidence"
