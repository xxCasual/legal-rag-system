"""Lightweight checks for Agent routing and tool selection."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app.agent.graph as agent_graph  # noqa: E402


def test_policy_router_prefers_company_document_hints() -> None:
    result = agent_graph.intent_router({"query": "公司的工资发放日是什么时候？"})
    assert result["intent"] == "policy_qa"


def test_policy_router_uses_internal_policy_hint_without_llm() -> None:
    result = agent_graph.intent_router(
        {"query": "报销 30 天这个限制是劳动法要求的吗？我主要要查公司内部有没有这个期限。"}
    )
    assert result["intent"] == "policy_qa"
    assert result["intent_source"] == "rule"


def test_contract_router_prefers_contract_review_hints() -> None:
    result = agent_graph.intent_router(
        {"query": "请审查这份劳动合同：甲方可随时解除合同且不支付补偿。"}
    )
    assert result["intent"] == "contract_review"


def test_contract_router_takes_precedence_over_policy_words_in_contract_text() -> None:
    result = agent_graph.intent_router(
        {
            "query": (
                "合同片段：劳动合同期限为三年，试用期三个月，工资另行约定，"
                "社保缴纳方式按公司内部政策执行，解除条件按员工手册处理。"
            )
        }
    )
    assert result["intent"] == "contract_review"


def test_refusal_router_uses_local_hints_without_llm() -> None:
    result = agent_graph.intent_router(
        {"query": "帮我写一份员工辞退通知，理由随便写严重点，最好让公司不用赔钱。"}
    )
    assert result["intent"] == "refusal"
    assert result["intent_source"] == "rule"


def test_law_router_uses_local_classifier_metadata() -> None:
    result = agent_graph.intent_router({"query": "试用期最长多久？"})

    assert result["intent"] == "law_qa"
    assert result["intent_source"] == "rule"
    assert result["intent_confidence"] == 1.0


def test_law_router_uses_local_law_hints_without_llm() -> None:
    queries = [
        "员工入职后，公司最晚什么时候必须和他签书面劳动合同？",
        "劳动合同到期不续签，公司一定要给经济补偿吗？",
        "试用期最长能约定多久？两年的合同可以约定 6 个月试用期吗？",
        "员工主动辞职，需要提前多少天通知公司？试用期是不是不一样？",
        "公司安排员工周末加班，法律上加班费怎么算？",
        "员工医疗期内，公司能不能解除劳动合同？",
        "公司未缴社保，员工能不能以此为理由解除劳动合同并要求补偿？",
        "女员工怀孕期间绩效不好，公司可以辞退吗？",
        "公司单方面调岗降薪，员工不同意的话法律上怎么处理？",
        "劳动仲裁的时效一般是多久？从什么时候开始算？",
        "加班费法律怎么规定？如果员工没有主管提前审批，能不能算加班？",
        "公司制度写加班要审批，那如果法律上员工确实加班了，公司还能拒绝加班费吗？",
    ]
    results = [agent_graph.intent_router({"query": query}) for query in queries]

    assert [result["intent"] for result in results] == ["law_qa"] * len(queries)


def test_route_after_intent_maps_all_supported_intents() -> None:
    assert agent_graph.route_after_intent({"intent": "law_qa"}) == "law_qa_node"
    assert agent_graph.route_after_intent({"intent": "policy_qa"}) == "policy_qa_node"
    assert agent_graph.route_after_intent({"intent": "contract_review"}) == "contract_review_node"
    assert agent_graph.route_after_intent({"intent": "refusal"}) == "refusal_node"


def test_policy_node_selects_policy_tool() -> None:
    original_tool = agent_graph.search_company_policy
    agent_graph.search_company_policy = lambda query: {
        "answer": f"{query} - 制度答案",
        "citations": ["制度片段"],
        "contexts": ["制度片段"],
        "route": "policy_qa",
    }
    try:
        result = agent_graph.policy_qa_node(
            {"query": "公司的工资发放日是什么时候？", "intent": "policy_qa"}
        )
    finally:
        agent_graph.search_company_policy = original_tool

    assert result["tools_used"] == ["search_company_policy"]
    assert result["citations"] == ["制度片段"]
    assert result["route"] == "policy_qa"


def test_contract_node_selects_contract_review_tool() -> None:
    original_tool = agent_graph.review_labor_contract
    agent_graph.review_labor_contract = lambda query: {
        "answer": "合同审查摘要",
        "citations": [],
        "contexts": [],
        "route": "contract_review",
        "result_type": "contract_review",
        "risk_level": "medium",
        "review_status": "not_required",
        "contract_review": {"risk_level": "medium"},
    }
    try:
        result = agent_graph.contract_review_node(
            {"query": "请审查劳动合同", "intent": "contract_review"}
        )
    finally:
        agent_graph.review_labor_contract = original_tool

    assert result["tools_used"] == ["contract_review_rules"]
    assert result["risk_level"] == "medium"
    assert result["contract_review"] == {"risk_level": "medium"}


def test_refusal_node_marks_result_as_refused() -> None:
    result = agent_graph.refusal_node({"query": "今天天气怎么样？", "intent": "refusal"})

    assert result["tools_used"] == []
    assert result["route"] == "refusal"
    assert result["result_type"] == "refusal"
    assert result["retrieval_payload"]["refused"] is True


if __name__ == "__main__":
    test_policy_router_prefers_company_document_hints()
    test_policy_router_uses_internal_policy_hint_without_llm()
    test_contract_router_prefers_contract_review_hints()
    test_contract_router_takes_precedence_over_policy_words_in_contract_text()
    test_refusal_router_uses_local_hints_without_llm()
    test_law_router_uses_local_classifier_metadata()
    test_law_router_uses_local_law_hints_without_llm()
    test_route_after_intent_maps_all_supported_intents()
    test_policy_node_selects_policy_tool()
    test_contract_node_selects_contract_review_tool()
    test_refusal_node_marks_result_as_refused()
    print("agent graph ok")
