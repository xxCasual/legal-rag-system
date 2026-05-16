"""Tool wrappers used by the LangGraph assistant workflow."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.document_service import document_service
from app.services.rag_service import rag_service


def search_law_articles(query: str) -> Dict[str, Any]:
    result = rag_service.chat(query)
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "route": result["route"],
        "contexts": result["citations"],
        "result_type": "law_qa",
    }


def search_company_policy(query: str) -> Dict[str, Any]:
    docs = document_service.search(query, k=4)
    contexts = [doc.page_content for doc in docs]
    answer = _answer_policy_question(query, contexts)
    return {
        "answer": answer,
        "citations": contexts,
        "route": "policy_qa",
        "contexts": contexts,
        "result_type": "policy_qa",
    }


def review_labor_contract(
    contract_text: str,
    include_evidence: bool = False,
) -> Dict[str, Any]:
    from app.services.contract_review_service import contract_review_service
    from app.services.review_service import review_service

    result = contract_review_service.review_contract(
        contract_text,
        include_evidence=include_evidence,
    )
    risk_level = result.get("risk_level", "low")
    if risk_level == "high":
        pending = review_service.create_pending_review(
            source_type="contract_review",
            payload=_contract_review_pending_payload(contract_text, result),
            final_answer=result,
        )
        review_id = pending["review_id"]
        summary = "高风险合同审查结果已进入人工复核队列，审批通过后再返回完整审查结果。"
        contract_review = {
            "risk_level": "high",
            "findings": [],
            "evidence": [],
            "suggestions": [summary],
            "disclaimer": result.get("disclaimer", "仅供参考，需人工复核"),
            "latency": result.get("latency", 0),
            "review_status": "pending_review",
            "review_id": review_id,
        }
        return {
            "answer": summary,
            "citations": [],
            "contexts": [],
            "route": "contract_review",
            "result_type": "contract_review",
            "risk_level": "high",
            "review_status": "pending_review",
            "review_id": review_id,
            "contract_review": contract_review,
        }

    answer = _contract_review_answer(result)
    return {
        "answer": answer,
        "citations": result.get("evidence", []),
        "contexts": result.get("evidence", []),
        "route": "contract_review",
        "result_type": "contract_review",
        "risk_level": risk_level,
        "review_status": "not_required",
        "review_id": None,
        "contract_review": {
            **result,
            "review_status": "not_required",
            "review_id": None,
        },
    }


def refuse_out_of_scope(query: str) -> Dict[str, Any]:
    answer = (
        "超出范围，无法提供该请求的帮助。这个系统只能回答中国劳动合规、企业制度和劳动合同审查相关问题。"
        "你可以问我试用期、工资、加班、解除劳动合同或企业制度相关问题。"
    )
    return {
        "answer": answer,
        "citations": [],
        "contexts": [],
        "route": "refusal",
        "result_type": "refusal",
        "refused": True,
    }


def _contract_review_answer(result: Dict[str, Any]) -> str:
    risk_level = result.get("risk_level", "low")
    findings = result.get("findings", [])
    risky_clauses = [
        finding.get("clause_name", "")
        for finding in findings
        if finding.get("risk_level") in {"medium", "high"}
    ]
    if risky_clauses:
        clauses = "、".join(clause for clause in risky_clauses if clause)
        return f"合同审查完成，整体风险等级为 {risk_level}。建议重点复核：{clauses}。"
    return f"合同审查完成，整体风险等级为 {risk_level}，暂未发现中高风险条款。"


def _contract_review_pending_payload(contract_text: str, result: Dict[str, Any]) -> Dict[str, Any]:
    high_risk_clauses = [
        finding["clause_name"]
        for finding in result.get("findings", [])
        if finding.get("risk_level") == "high"
    ]
    preview = contract_text.strip()
    return {
        "risk_level": result.get("risk_level", "high"),
        "source": "labor_contract_review",
        "summary": "高风险劳动合同审查结果待人工复核",
        "high_risk_clauses": high_risk_clauses,
        "contract_text_preview": preview[:500],
    }


def _answer_policy_question(query: str, contexts: List[str]) -> str:
    if not contexts:
        return "当前企业制度文档中未检索到相关依据。"

    prompt = ChatPromptTemplate.from_template(
        """你是企业劳动合规助手。请严格根据以下企业制度文档回答问题。

要求：
1. 只根据提供的企业制度回答，不要补充外部知识
2. 如果制度中没有明确依据，请回答“当前企业制度文档中未检索到相关依据”
3. 回答简洁清楚

企业制度文档：
{context}

用户问题：{question}"""
    )
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        timeout=30,
        max_retries=0,
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "context": "\n\n".join(contexts),
            "question": query,
        }
    )
