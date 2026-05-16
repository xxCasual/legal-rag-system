"""Local intent classification for the Agent graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from app.core.config import settings


SUPPORTED_INTENTS = {"law_qa", "policy_qa", "contract_review", "refusal"}

DEFAULT_PROTOTYPES: Mapping[str, Sequence[str]] = {
    "law_qa": ("法律规定、赔偿、工资、加班、仲裁、劳动合同法问题",),
    "policy_qa": ("公司制度、员工手册、报销、考勤、请假、内部流程问题",),
    "contract_review": ("审查劳动合同文本、识别合同条款风险",),
    "refusal": ("无关问题、违法违规请求、天气股票编程闲聊",),
}

CONTRACT_REVIEW_HINTS = (
    "合同审查",
    "审查合同",
    "合同风险",
    "风险审查",
    "合同文本",
    "合同片段",
    "合同条款",
    "合同写",
    "请审查",
    "请判断风险等级",
    "帮看风险",
    "帮忙看合同风险",
    "法务帮忙看",
    "审一下",
    "帮我审",
    "审查一下",
    "劳动合同期限",
    "本合同",
    "甲方",
    "乙方",
    "员工自愿",
    "放弃社保",
    "不支付补偿",
    "无需支付竞业限制补偿",
)

REFUSAL_HINTS = (
    "不用赔钱",
    "不被发现",
    "理由随便写严重点",
    "偷偷",
    "摄像头监控",
    "监控员工",
    "电脑屏幕",
    "聊天记录",
    "破解",
    "密码",
    "股票",
    "天气",
    "信佛",
    "筛掉",
    "不要录用",
)

POLICY_PRIORITY_HINTS = (
    "主要想确认公司制度",
    "主要要查公司内部",
    "先按我们制度",
    "先查一下我们公司",
    "不是问劳动法",
    "我不是问劳动法",
    "问公司规定",
    "我们内部是不是",
    "内部有没有",
    "按公司制度要怎么处理",
    "公司制度认不认",
    "按制度还能",
    "制度上需要",
    "公司规定是多久",
    "考勤制度",
    "请假制度",
    "年假申请",
    "请年假",
    "直属主管",
    "提前多久提交",
    "临时申请",
    "医院证明",
    "病假材料",
    "绩效考核",
    "工资到底是几号发",
    "工资发放日",
    "几号发",
)

LAW_HINTS = (
    "劳动法",
    "劳动合同法",
    "劳动合同",
    "书面劳动合同",
    "签书面",
    "入职",
    "到期不续签",
    "试用期",
    "主动辞职",
    "提前多少天通知",
    "工资",
    "降薪",
    "调岗",
    "加班费",
    "法律上",
    "法律怎么规定",
    "社保",
    "社会保险",
    "未缴社保",
    "工伤",
    "仲裁",
    "仲裁时效",
    "辞退",
    "解雇",
    "解除劳动合同",
    "医疗期",
    "怀孕",
    "经济补偿",
    "补偿",
    "赔偿",
)

POLICY_HINTS = (
    "员工手册",
    "公司规定",
    "企业制度",
    "内部制度",
    "内部规定",
    "内部政策",
    "公司内部",
    "公司制度",
    "考勤",
    "请假",
    "病假",
    "年假",
    "绩效",
    "报销制度",
    "报销",
    "工资发放日",
    "迟到",
    "打卡",
)


@dataclass(frozen=True)
class IntentResult:
    intent: str
    confidence: float
    source: str


class IntentClassifier:
    """Classify Agent intent with deterministic rules and local embeddings."""

    def __init__(
        self,
        *,
        embedder: object | None = None,
        prototypes: Mapping[str, Sequence[str]] | None = None,
        enable_embedding_fallback: bool = True,
        embedding_threshold: float = 0.46,
        embedding_model: str | None = None,
    ) -> None:
        self._embedder = embedder
        self._prototypes = prototypes or DEFAULT_PROTOTYPES
        self._enable_embedding_fallback = enable_embedding_fallback
        self._embedding_threshold = embedding_threshold
        self._embedding_model = embedding_model or settings.embedding_model
        self._prototype_vectors: list[tuple[str, str, Sequence[float]]] | None = None

    def classify(self, query: str) -> IntentResult:
        normalized = query.strip()
        if not normalized:
            return IntentResult("refusal", 1.0, "empty")

        rule_result = self._rule_based_intent(normalized)
        if rule_result is not None:
            return rule_result

        if self._enable_embedding_fallback:
            embedding_result = self._embedding_intent(normalized)
            if embedding_result is not None:
                return embedding_result

        return IntentResult("refusal", 0.0, "low_confidence")

    def _rule_based_intent(self, query: str) -> IntentResult | None:
        if _contains_any(query, CONTRACT_REVIEW_HINTS):
            return IntentResult("contract_review", 1.0, "rule")
        if _contains_any(query, REFUSAL_HINTS):
            return IntentResult("refusal", 1.0, "rule")
        if _contains_any(query, POLICY_PRIORITY_HINTS):
            return IntentResult("policy_qa", 1.0, "rule")
        if _contains_any(query, LAW_HINTS):
            return IntentResult("law_qa", 1.0, "rule")
        if _contains_any(query, POLICY_HINTS):
            return IntentResult("policy_qa", 1.0, "rule")
        return None

    def _embedding_intent(self, query: str) -> IntentResult | None:
        try:
            embedder = self._get_embedder()
            query_vector = _as_vector(embedder.encode([query], normalize_embeddings=True)[0])
            best_intent = "refusal"
            best_score = -1.0
            for intent, _, prototype_vector in self._get_prototype_vectors(embedder):
                score = _cosine(query_vector, prototype_vector)
                if score > best_score:
                    best_intent = intent
                    best_score = score
        except Exception:
            return None

        if best_score < self._embedding_threshold:
            return IntentResult("refusal", best_score, "low_confidence")
        return IntentResult(best_intent, best_score, "embedding")

    def _get_embedder(self):  # noqa: ANN202
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._embedding_model)
        return self._embedder

    def _get_prototype_vectors(
        self, embedder: object
    ) -> list[tuple[str, str, Sequence[float]]]:
        if self._prototype_vectors is None:
            texts = [
                prototype
                for prototypes in self._prototypes.values()
                for prototype in prototypes
            ]
            encoded = embedder.encode(texts, normalize_embeddings=True)
            vectors = [_as_vector(vector) for vector in encoded]
            self._prototype_vectors = []
            index = 0
            for intent, prototypes in self._prototypes.items():
                for prototype in prototypes:
                    self._prototype_vectors.append((intent, prototype, vectors[index]))
                    index += 1
        return self._prototype_vectors


def _contains_any(query: str, keywords: Sequence[str]) -> bool:
    return any(keyword in query for keyword in keywords)


def _as_vector(vector: object) -> Sequence[float]:
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return vector  # type: ignore[return-value]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
