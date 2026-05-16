"""Lightweight checks for labor contract risk review."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.services.contract_review_service as review_module  # noqa: E402
from app.services.contract_review_service import (  # noqa: E402
    CLAUSE_SPECS,
    DISCLAIMER,
    ContractReviewService,
)


def _patch_law_search():
    calls = []

    def fake_search(query: str):
        calls.append(query)
        return {
            "citations": [f"引用-{query}", "共同依据", "共同依据"],
            "contexts": [f"引用-{query}", "共同依据", "共同依据"],
            "route": "law_qa",
        }

    return calls, fake_search


def test_review_calls_law_search_for_each_clause_and_flags_high_risk() -> None:
    calls, fake_search = _patch_law_search()
    service = ContractReviewService(law_search=fake_search)
    text = """
    第一条 合同期限为三年。
    第二条 试用期一年。
    第三条 工资为每月八千元。
    第四条 执行标准工时，员工自愿加班且不支付加班费。
    第五条 员工自愿放弃社保。
    第六条 甲方可随时解除劳动合同且不支付经济补偿。
    第七条 员工承担竞业限制义务但不支付补偿。
    """

    result = service.review_contract(text)

    findings = {finding["clause_type"]: finding for finding in result["findings"]}
    assert len(calls) == len(CLAUSE_SPECS)
    assert result["risk_level"] == "high"
    assert result["disclaimer"] == DISCLAIMER
    assert findings["probation"]["risk_level"] == "high"
    assert findings["social_insurance"]["risk_level"] == "high"
    assert findings["termination"]["risk_level"] == "high"
    assert findings["non_compete"]["risk_level"] == "high"
    assert result["evidence"].count("共同依据") == 1


def test_review_marks_missing_required_clauses_as_medium() -> None:
    calls, fake_search = _patch_law_search()
    service = ContractReviewService(law_search=fake_search)

    result = service.review_contract("双方签订劳动关系文件。试用期一个月。")

    findings = {finding["clause_type"]: finding for finding in result["findings"]}
    assert len(calls) == len(CLAUSE_SPECS)
    assert result["risk_level"] == "medium"
    assert findings["salary"]["status"] == "missing"
    assert findings["salary"]["risk_level"] == "medium"
    assert findings["social_insurance"]["risk_level"] == "medium"
    assert findings["non_compete"]["status"] == "missing"
    assert findings["non_compete"]["risk_level"] == "low"


def test_review_can_skip_law_evidence_for_fast_eval() -> None:
    calls, fake_search = _patch_law_search()
    service = ContractReviewService(law_search=fake_search)

    result = service.review_contract(
        "合同期限为三年。试用期一年。员工自愿放弃社保。",
        include_evidence=False,
    )

    assert calls == []
    assert result["risk_level"] == "high"
    assert result["evidence"] == []
    assert all(finding["evidence"] == [] for finding in result["findings"])


def test_social_insurance_cash_substitute_is_high_risk() -> None:
    calls, fake_search = _patch_law_search()
    service = ContractReviewService(law_search=fake_search)

    result = service.review_contract(
        "合同期限三年。工资每月一万元。执行标准工时。"
        "双方约定社会保险由公司每月支付现金补贴替代，员工自行处理。",
        include_evidence=False,
    )

    findings = {finding["clause_type"]: finding for finding in result["findings"]}
    assert calls == []
    assert findings["social_insurance"]["risk_level"] == "high"
    assert result["risk_level"] == "high"


def test_working_hours_no_extra_overtime_pay_is_high_risk() -> None:
    calls, fake_search = _patch_law_search()
    service = ContractReviewService(law_search=fake_search)

    result = service.review_contract(
        "合同期限三年。工资每月一万元。公司依法缴纳社会保险。"
        "乙方加班的，公司不另行支付任何加班费。",
        include_evidence=False,
    )

    findings = {finding["clause_type"]: finding for finding in result["findings"]}
    assert calls == []
    assert findings["working_hours"]["risk_level"] == "high"
    assert result["risk_level"] == "high"


def test_non_compete_without_compensation_is_high_risk() -> None:
    calls, fake_search = _patch_law_search()
    service = ContractReviewService(law_search=fake_search)

    result = service.review_contract(
        "合同期限三年。工资每月一万元。公司依法缴纳社会保险。"
        "乙方离职后承担竞业限制义务，甲方无需支付任何竞业补偿。",
        include_evidence=False,
    )

    findings = {finding["clause_type"]: finding for finding in result["findings"]}
    assert calls == []
    assert findings["non_compete"]["risk_level"] == "high"
    assert result["risk_level"] == "high"


if __name__ == "__main__":
    test_review_calls_law_search_for_each_clause_and_flags_high_risk()
    test_review_marks_missing_required_clauses_as_medium()
    test_review_can_skip_law_evidence_for_fast_eval()
    test_social_insurance_cash_substitute_is_high_risk()
    test_working_hours_no_extra_overtime_pay_is_high_risk()
    test_non_compete_without_compensation_is_high_risk()
    print("contract review service ok")
