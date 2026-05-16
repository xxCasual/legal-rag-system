"""LangGraph assistant workflow for legal QA, policy QA, contract review, and refusal."""

from __future__ import annotations

import time
from typing import Dict

from langgraph.graph import END, START, StateGraph

from app.agent.intent_classifier import SUPPORTED_INTENTS, IntentClassifier
from app.agent.state import AgentState
from app.agent.tools import (
    refuse_out_of_scope,
    review_labor_contract,
    search_company_policy,
    search_law_articles,
)


intent_classifier = IntentClassifier()


def intent_router(state: AgentState) -> AgentState:
    query = state["query"]
    result = intent_classifier.classify(query)
    intent = result.intent
    if intent not in SUPPORTED_INTENTS:
        intent = "refusal"
    return {
        "intent": intent,
        "route": intent,
        "intent_source": result.source,
        "intent_confidence": round(result.confidence, 4),
    }


def route_after_intent(state: AgentState) -> str:
    return {
        "law_qa": "law_qa_node",
        "policy_qa": "policy_qa_node",
        "contract_review": "contract_review_node",
        "refusal": "refusal_node",
    }.get(state.get("intent", "law_qa"), "law_qa_node")


def law_qa_node(state: AgentState) -> AgentState:
    payload = search_law_articles(state["query"])
    return _state_from_payload(payload, ["search_law_articles"])


def policy_qa_node(state: AgentState) -> AgentState:
    payload = search_company_policy(state["query"])
    return _state_from_payload(payload, ["search_company_policy"])


def contract_review_node(state: AgentState) -> AgentState:
    payload = review_labor_contract(state["query"])
    return _state_from_payload(payload, ["contract_review_rules"])


def refusal_node(state: AgentState) -> AgentState:
    payload = refuse_out_of_scope(state["query"])
    return _state_from_payload(payload, [])


def _state_from_payload(payload: Dict[str, object], tools_used: list[str]) -> AgentState:
    return {
        "retrieval_payload": payload,
        "retrieved_contexts": payload.get("contexts", []),
        "citations": payload.get("citations", []),
        "tools_used": tools_used,
        "route": payload.get("route", ""),
        "result_type": payload.get("result_type", payload.get("route", "")),
        "answer": payload.get("answer", ""),
        "risk_level": payload.get("risk_level"),
        "review_status": payload.get("review_status"),
        "review_id": payload.get("review_id"),
        "contract_review": payload.get("contract_review"),
    }


def build_agent_graph():
    builder = StateGraph(AgentState)
    builder.add_node("intent_router", intent_router)
    builder.add_node("law_qa_node", law_qa_node)
    builder.add_node("policy_qa_node", policy_qa_node)
    builder.add_node("contract_review_node", contract_review_node)
    builder.add_node("refusal_node", refusal_node)
    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "law_qa_node": "law_qa_node",
            "policy_qa_node": "policy_qa_node",
            "contract_review_node": "contract_review_node",
            "refusal_node": "refusal_node",
        },
    )
    builder.add_edge("law_qa_node", END)
    builder.add_edge("policy_qa_node", END)
    builder.add_edge("contract_review_node", END)
    builder.add_edge("refusal_node", END)
    return builder.compile()


agent_graph = build_agent_graph()


def run_agent_chat(query: str) -> Dict[str, object]:
    started_at = time.perf_counter()
    result = agent_graph.invoke({"query": query})
    latency = round(time.perf_counter() - started_at, 3)
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "route": result.get("route", result.get("intent", "")),
        "intent": result.get("intent", ""),
        "intent_source": result.get("intent_source", ""),
        "intent_confidence": result.get("intent_confidence", 0.0),
        "tools_used": result.get("tools_used", []),
        "result_type": result.get("result_type", result.get("route", "")),
        "risk_level": result.get("risk_level"),
        "review_status": result.get("review_status"),
        "review_id": result.get("review_id"),
        "contract_review": result.get("contract_review"),
        "latency": latency,
    }
