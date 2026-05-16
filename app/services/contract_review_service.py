"""Rule-based labor contract risk review service."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence


DISCLAIMER = "仅供参考，需人工复核"

RISK_ORDER = {"low": 0, "medium": 1, "high": 2}
REQUIRED_CLAUSES = {"term", "salary", "working_hours", "social_insurance"}
LawSearch = Callable[[str], Dict[str, Any]]


@dataclass(frozen=True)
class ClauseSpec:
    clause_type: str
    clause_name: str
    keywords: Sequence[str]
    search_query: str


CLAUSE_SPECS: Sequence[ClauseSpec] = (
    ClauseSpec(
        clause_type="probation",
        clause_name="试用期",
        keywords=("试用期", "试用"),
        search_query="劳动合同 试用期 法律要求 风险审查",
    ),
    ClauseSpec(
        clause_type="term",
        clause_name="合同期限",
        keywords=("合同期限", "合同期", "固定期限", "无固定期限", "起始日期", "终止日期"),
        search_query="劳动合同 合同期限 法律要求 风险审查",
    ),
    ClauseSpec(
        clause_type="salary",
        clause_name="工资",
        keywords=("工资", "薪资", "薪酬", "劳动报酬", "报酬"),
        search_query="劳动合同 工资 劳动报酬 法律要求 风险审查",
    ),
    ClauseSpec(
        clause_type="working_hours",
        clause_name="工时",
        keywords=("工时", "工作时间", "加班", "休息休假", "标准工时", "综合工时"),
        search_query="劳动合同 工时 工作时间 加班 法律要求 风险审查",
    ),
    ClauseSpec(
        clause_type="social_insurance",
        clause_name="社保",
        keywords=("社保", "社会保险", "五险", "养老保险", "医疗保险", "失业保险", "工伤保险", "生育保险"),
        search_query="劳动合同 社会保险 社保 法律要求 风险审查",
    ),
    ClauseSpec(
        clause_type="termination",
        clause_name="解除",
        keywords=("解除", "终止", "辞退", "离职", "经济补偿", "赔偿"),
        search_query="劳动合同 解除 终止 经济补偿 法律要求 风险审查",
    ),
    ClauseSpec(
        clause_type="non_compete",
        clause_name="竞业限制",
        keywords=("竞业限制", "竞业禁止", "竞业", "保密义务"),
        search_query="劳动合同 竞业限制 经济补偿 法律要求 风险审查",
    ),
)


class ContractReviewService:
    """Review labor contract text with rules plus legal RAG evidence."""

    def __init__(self, law_search: LawSearch | None = None) -> None:
        self._law_search = law_search or _default_law_search

    def review_contract(
        self,
        contract_text: str,
        include_evidence: bool = True,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()
        cleaned_text = contract_text.strip()
        if not cleaned_text:
            raise ValueError("contract_text must not be blank")

        paragraphs = self._split_paragraphs(cleaned_text)
        findings: List[Dict[str, Any]] = []
        all_evidence: List[str] = []
        suggestions: List[str] = []

        for spec in CLAUSE_SPECS:
            extracted_text = self._extract_clause(paragraphs, spec)
            evidence = self._law_evidence(spec) if include_evidence else []
            finding = self._build_finding(spec, extracted_text, evidence)
            findings.append(finding)
            all_evidence.extend(evidence)
            suggestions.append(finding["suggestion"])

        latency = round(time.perf_counter() - started_at, 3)
        return {
            "risk_level": self._overall_risk(findings),
            "findings": findings,
            "evidence": self._dedupe(all_evidence),
            "suggestions": self._dedupe(suggestions),
            "disclaimer": DISCLAIMER,
            "latency": latency,
        }

    def _law_evidence(self, spec: ClauseSpec) -> List[str]:
        legal_payload = self._law_search(spec.search_query)
        return self._evidence_from_payload(legal_payload)

    def _split_paragraphs(self, text: str) -> List[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        blocks = re.split(r"\n\s*\n|\n|(?=第[一二三四五六七八九十百\d]+条)", normalized)
        paragraphs = [block.strip() for block in blocks if block.strip()]
        return paragraphs or [normalized.strip()]

    def _extract_clause(self, paragraphs: Sequence[str], spec: ClauseSpec) -> str:
        snippets: List[str] = []
        for paragraph in paragraphs:
            for keyword in spec.keywords:
                position = paragraph.find(keyword)
                if position >= 0:
                    snippets.append(self._context_snippet(paragraph, position))
                    break
            if len(snippets) >= 3:
                break
        return "\n".join(self._dedupe(snippets))

    def _context_snippet(self, text: str, position: int, before: int = 120, after: int = 360) -> str:
        start = max(position - before, 0)
        end = min(position + after, len(text))
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    def _evidence_from_payload(self, payload: Dict[str, Any]) -> List[str]:
        raw_items = payload.get("contexts") or payload.get("citations") or []
        evidence = [self._truncate(str(item).strip(), 800) for item in raw_items if str(item).strip()]
        return self._dedupe(evidence)[:2]

    def _build_finding(
        self,
        spec: ClauseSpec,
        extracted_text: str,
        evidence: List[str],
    ) -> Dict[str, Any]:
        status = self._clause_status(extracted_text)
        risk_level = self._risk_level(spec, extracted_text, status)
        return {
            "clause_type": spec.clause_type,
            "clause_name": spec.clause_name,
            "status": status,
            "risk_level": risk_level,
            "extracted_text": extracted_text,
            "analysis": self._analysis(spec, status, risk_level),
            "evidence": evidence,
            "suggestion": self._suggestion(spec, status, risk_level),
        }

    def _clause_status(self, extracted_text: str) -> str:
        if not extracted_text:
            return "missing"
        unclear_terms = (
            "另行约定",
            "另行确定",
            "按公司规定",
            "按甲方规定",
            "视情况",
            "待定",
            "不明确",
            "具体另定",
            "公司解释",
            "甲方解释",
        )
        return "unclear" if any(term in extracted_text for term in unclear_terms) else "present"

    def _risk_level(self, spec: ClauseSpec, extracted_text: str, status: str) -> str:
        if status == "present" and self._has_high_risk_pattern(spec, extracted_text):
            return "high"
        if status == "unclear":
            return "medium"
        if status == "missing" and spec.clause_type in REQUIRED_CLAUSES:
            return "medium"
        return "low"

    def _has_high_risk_pattern(self, spec: ClauseSpec, text: str) -> bool:
        if spec.clause_type == "probation":
            months = self._probation_months(text)
            return months is not None and months > 6
        if spec.clause_type == "social_insurance":
            return self._contains_any(
                text,
                (
                    "放弃社保",
                    "放弃社会保险",
                    "不缴纳社保",
                    "不缴纳社会保险",
                    "无需缴纳社保",
                    "不予缴纳社保",
                    "自行承担社保",
                    "现金补贴代替社保",
                    "社保补贴替代",
                ),
            ) or self._matches_any(
                text,
                (
                    r"(社保|社会保险)[^。；;\n]{0,50}(现金补贴|补贴)[^。；;\n]{0,20}(替代|代替)",
                    r"(现金补贴|补贴)[^。；;\n]{0,20}(替代|代替)[^。；;\n]{0,50}(社保|社会保险)",
                    r"放弃[^。；;\n]{0,30}(缴纳|购买|参加)[^。；;\n]{0,30}(社保|社会保险)",
                    r"(员工|乙方)[^。；;\n]{0,20}自行[^。；;\n]{0,20}(缴纳|处理)[^。；;\n]{0,30}(社保|社会保险)",
                ),
            )
        if spec.clause_type == "non_compete":
            has_non_compete = self._contains_any(text, ("竞业限制", "竞业禁止", "竞业"))
            no_compensation = self._contains_any(
                text,
                ("无补偿", "不支付补偿", "无需补偿", "不予补偿"),
            ) or self._matches_any(
                text,
                (
                    r"(无需|不予|不再|不另行|不)支付[^。；;\n]{0,20}(竞业)?补偿",
                    r"(竞业限制|竞业禁止|竞业)[^。；;\n]{0,60}(无|没有|无需)[^。；;\n]{0,20}补偿",
                ),
            )
            has_compensation = self._contains_any(text, ("补偿", "经济补偿", "竞业补偿", "按月支付"))
            return has_non_compete and (no_compensation or not has_compensation)
        if spec.clause_type == "termination":
            return self._contains_any(
                text,
                (
                    "随时解除",
                    "无条件解除",
                    "无需补偿",
                    "不支付经济补偿",
                    "不予经济补偿",
                    "甲方可随时",
                    "甲方有权随时",
                ),
            )
        if spec.clause_type == "working_hours":
            return self._contains_any(
                text,
                (
                    "加班不支付加班费",
                    "不支付加班费",
                    "无加班费",
                    "加班费已包含",
                    "工资包含全部加班费",
                    "自愿加班",
                    "不计加班",
                ),
            ) or self._matches_any(
                text,
                (
                    r"不(再|另行)?支付[^。；;\n]{0,20}加班费",
                    r"加班[^。；;\n]{0,40}不(再|另行)?支付[^。；;\n]{0,20}(费用|报酬)",
                ),
            )
        if spec.clause_type == "salary":
            return self._contains_any(text, ("无工资", "不支付工资", "工资待定", "低于最低工资"))
        return False

    def _probation_months(self, text: str) -> int | None:
        for match in re.finditer(r"试用期[^。；;\n]{0,20}?([一二三四五六七八九十\d]+)\s*(个月|月|年)", text):
            number = self._parse_chinese_number(match.group(1))
            if number is None:
                continue
            unit = match.group(2)
            return number * 12 if unit == "年" else number
        return None

    def _parse_chinese_number(self, value: str) -> int | None:
        if value.isdigit():
            return int(value)
        digits = {
            "零": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if value in digits:
            return digits[value]
        if value == "十":
            return 10
        if "十" in value:
            left, _, right = value.partition("十")
            tens = digits.get(left, 1) if left else 1
            ones = digits.get(right, 0) if right else 0
            return tens * 10 + ones
        return None

    def _analysis(self, spec: ClauseSpec, status: str, risk_level: str) -> str:
        if risk_level == "high":
            return f"{spec.clause_name}条款可能存在较高合规风险，建议结合检索依据进行人工复核。"
        if status == "missing":
            return f"未检出明确的{spec.clause_name}条款，可能影响合同履行和争议处理，建议人工复核。"
        if status == "unclear":
            return f"{spec.clause_name}条款表述可能不够明确，可能存在解释争议，建议人工复核。"
        return f"已检出{spec.clause_name}条款，暂未触发明显高风险规则，仍建议人工复核。"

    def _suggestion(self, spec: ClauseSpec, status: str, risk_level: str) -> str:
        if risk_level == "high":
            return f"请重点复核{spec.clause_name}条款，删除或调整可能违法、不合理的表述，并由人工确认。"
        if status == "missing" and spec.clause_type in REQUIRED_CLAUSES:
            return f"建议补充明确的{spec.clause_name}条款，包括适用范围、标准和履行方式。"
        if status == "unclear":
            return f"建议将{spec.clause_name}条款改写为明确、可执行的约定，避免仅写“按公司规定”或“另行约定”。"
        if status == "missing":
            return f"未检出{spec.clause_name}条款，暂未直接判定为高风险；如业务需要，建议人工确认是否补充。"
        return f"建议保留{spec.clause_name}条款的明确表述，并在签署前进行人工复核。"

    def _overall_risk(self, findings: Sequence[Dict[str, Any]]) -> str:
        return max((finding["risk_level"] for finding in findings), key=lambda risk: RISK_ORDER[risk])

    def _contains_any(self, text: str, patterns: Iterable[str]) -> bool:
        return any(pattern in text for pattern in patterns)

    def _matches_any(self, text: str, patterns: Iterable[str]) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)

    def _truncate(self, text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: limit - 3] + "..."

    def _dedupe(self, items: Iterable[str]) -> List[str]:
        seen = set()
        deduped: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped


def _default_law_search(query: str) -> Dict[str, Any]:
    from app.services.rag_service import rag_service

    result = rag_service.chat(query)
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "contexts": result.get("citations", []),
        "route": result.get("route", "law_qa"),
    }


contract_review_service = ContractReviewService()
