"""
Batch runner for Agent Eval subsets.

This script keeps the evaluation logic in evaluation.agent_eval and adds a
small orchestration layer for repeatable split/full runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from evaluation.agent_eval import (  # noqa: E402
    AGENT_RESULTS_DIR,
    AGENT_TESTSET_PATH,
    load_agent_testset,
    print_summary,
    run_agent_eval,
    save_results,
)


SUBSET_ORDER = ("law_qa", "policy_qa", "refusal", "contract_review")
SUITE_CHOICES = ("all", "split", "full", "law", "policy", "refusal", "contract")
SUITE_SUBSETS = {
    "all": (*SUBSET_ORDER, "full"),
    "split": SUBSET_ORDER,
    "full": ("full",),
    "law": ("law_qa",),
    "policy": ("policy_qa",),
    "refusal": ("refusal",),
    "contract": ("contract_review",),
}

COMPANY_POLICY_SEED_FILE_NAME = "agent_eval_company_policy_seed.txt"
COMPANY_POLICY_SEED_PROBE = "工资发放日 报销 迟到 加班审批"
COMPANY_POLICY_SEED_TEXT = """企业制度评测样例

1. 公司工资发放日为每月十日。
2. 员工连续迟到三次，应接受书面提醒。
3. 出差及日常费用报销应在费用发生后三十日内提交，超过期限需部门负责人特别批准。
4. 员工加班须提前通过系统提交申请并经直属主管审批；未提前审批的，原则上不认定为公司制度内加班。
5. 员工请年假需提前三个工作日提交申请，经直属主管批准后执行。
"""


def split_samples(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    return {
        "law_qa": [s for s in samples if s["expected_intent"] == "law_qa"],
        "policy_qa": [s for s in samples if s["expected_intent"] == "policy_qa"],
        "refusal": [
            s
            for s in samples
            if s["expected_tools"] == [] and s["expected_risk_level"] is None
        ],
        "contract_review": [
            s for s in samples if s["expected_risk_level"] is not None
        ],
        "full": list(samples),
    }


def selected_subsets(suite: str) -> Sequence[str]:
    if suite not in SUITE_SUBSETS:
        raise ValueError(f"Unsupported suite: {suite}")
    return SUITE_SUBSETS[suite]


def needs_company_policy_seed(subset_names: Sequence[str]) -> bool:
    return any(name in {"policy_qa", "full"} for name in subset_names)


def ensure_company_policy_seeded(document_service: Any | None = None) -> bool:
    if document_service is None:
        from app.services.document_service import document_service as default_service

        document_service = default_service

    existing = document_service.search(COMPANY_POLICY_SEED_PROBE, k=1)
    if existing:
        return False

    document_service.ingest_upload(
        COMPANY_POLICY_SEED_FILE_NAME,
        COMPANY_POLICY_SEED_TEXT.encode("utf-8"),
    )
    return True


def combined_tag(tag: str | None, subset_name: str) -> str:
    return f"{tag}_{subset_name}" if tag else subset_name


def average_latency(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get("latency") or 0) for row in rows) / len(rows), 3)


def summary_score(summary: Dict[str, Any], metric_name: str) -> str:
    score = summary["metrics"][metric_name]["score"]
    return "n/a" if score is None else f"{score:.4f}"


def build_suite_record(
    subset_name: str,
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    csv_path: Path,
    json_path: Path,
) -> Dict[str, Any]:
    return {
        "subset": subset_name,
        "samples": summary["total_samples"],
        "failed": summary["failed_samples"],
        "avg_latency": average_latency(rows),
        "metrics": summary["metrics"],
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


def write_suite_summary(
    records: Sequence[Dict[str, Any]],
    *,
    testset_path: Path,
    output_dir: Path,
    suite: str,
    tag: str | None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    path = output_dir / f"agent_eval{tag_part}_suite_{timestamp}.json"
    payload = {
        "testset_path": str(testset_path),
        "suite": suite,
        "tag": tag,
        "llm_model": settings.llm_model,
        "crag_mode": settings.crag_mode,
        "generated_at": timestamp,
        "runs": list(records),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_startup_info(
    *,
    testset_path: Path,
    output_dir: Path,
    suite: str,
    samples: Sequence[Dict[str, Any]],
    subsets: Dict[str, Sequence[Dict[str, Any]]],
) -> None:
    print("=" * 78)
    print("Agent Eval Suite")
    print("=" * 78)
    print(f"testset: {testset_path}")
    print(f"output_dir: {output_dir}")
    print(f"suite: {suite}")
    print(f"total_samples: {len(samples)}")
    for subset_name in (*SUBSET_ORDER, "full"):
        print(f"{subset_name}: {len(subsets[subset_name])}")
    print(f"LEGAL_RAG_LLM_MODEL: {settings.llm_model}")
    print(f"LEGAL_RAG_CRAG_MODE: {settings.crag_mode}")
    print("=" * 78)


def print_suite_table(records: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "subset",
        "samples",
        "failed",
        "intent",
        "tools",
        "refusal",
        "risk",
        "avg_latency",
        "csv_path",
    ]
    table_rows = []
    for record in records:
        summary = {
            "metrics": record["metrics"],
        }
        table_rows.append(
            [
                record["subset"],
                str(record["samples"]),
                str(record["failed"]),
                summary_score(summary, "intent_accuracy"),
                summary_score(summary, "tool_call_accuracy"),
                summary_score(summary, "refusal_accuracy"),
                summary_score(summary, "risk_accuracy"),
                f"{record['avg_latency']:.3f}s",
                record["csv_path"],
            ]
        )
    widths = [
        max(len(str(row[index])) for row in ([headers] + table_rows))
        for index in range(len(headers))
    ]

    print("\n" + "=" * 78)
    print("Agent Eval Suite Summary")
    print("=" * 78)
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
    print("=" * 78)


def contract_reviewer_with_optional_evidence(include_evidence: bool):
    if not include_evidence:
        return None

    from app.services.contract_review_service import contract_review_service

    return lambda text: contract_review_service.review_contract(
        text,
        include_evidence=True,
    )


def routing_only_agent_runner(query: str) -> Dict[str, Any]:
    """Evaluate routing/tool metrics without invoking RAG retrieval or answer LLMs."""
    import time

    from app.agent.intent_classifier import IntentClassifier
    from app.agent.tools import refuse_out_of_scope
    from app.services.contract_review_service import ContractReviewService

    started_at = time.perf_counter()
    intent_result = IntentClassifier(enable_embedding_fallback=False).classify(query)
    intent = intent_result.intent

    if intent == "contract_review":
        reviewer = ContractReviewService(law_search=lambda _: {"contexts": []})
        review = reviewer.review_contract(query, include_evidence=False)
        risk_level = review["risk_level"]
        review_status = "pending_review" if risk_level == "high" else "not_required"
        answer = (
            "高风险合同审查结果已进入人工复核队列。"
            if review_status == "pending_review"
            else f"合同审查完成，整体风险等级为 {risk_level}。"
        )
        return {
            "answer": answer,
            "citations": [],
            "route": "contract_review",
            "intent": "contract_review",
            "intent_source": intent_result.source,
            "intent_confidence": intent_result.confidence,
            "tools_used": ["contract_review_rules"],
            "result_type": "contract_review",
            "risk_level": risk_level,
            "review_status": review_status,
            "contract_review": review,
            "latency": round(time.perf_counter() - started_at, 3),
        }

    if intent == "refusal":
        payload = refuse_out_of_scope(query)
        return {
            "answer": payload["answer"],
            "citations": [],
            "route": "refusal",
            "intent": "refusal",
            "intent_source": intent_result.source,
            "intent_confidence": intent_result.confidence,
            "tools_used": [],
            "result_type": "refusal",
            "latency": round(time.perf_counter() - started_at, 3),
        }

    tools = {
        "law_qa": ["search_law_articles"],
        "policy_qa": ["search_company_policy"],
    }.get(intent, [])
    return {
        "answer": f"routing-only: {intent}",
        "citations": [],
        "route": intent,
        "intent": intent,
        "intent_source": intent_result.source,
        "intent_confidence": intent_result.confidence,
        "tools_used": tools,
        "result_type": intent,
        "latency": round(time.perf_counter() - started_at, 3),
    }


def run_suite(
    *,
    testset_path: Path,
    output_dir: Path,
    suite: str,
    tag: str | None,
    include_contract_evidence: bool,
    routing_only: bool = False,
) -> Path:
    samples = load_agent_testset(testset_path)
    subsets = split_samples(samples)
    subset_names = selected_subsets(suite)
    print_startup_info(
        testset_path=testset_path,
        output_dir=output_dir,
        suite=suite,
        samples=samples,
        subsets=subsets,
    )

    if needs_company_policy_seed(subset_names):
        if routing_only:
            print("\n>>> Routing-only mode: skipping company policy eval seed indexing")
        else:
            print("\n>>> Ensuring company policy eval seed is indexed...")
            seeded = ensure_company_policy_seeded()
            seed_status = "created" if seeded else "already available"
            print(f">>> Company policy eval seed: {seed_status}")

    contract_reviewer = contract_reviewer_with_optional_evidence(
        include_contract_evidence
    )
    agent_runner = routing_only_agent_runner if routing_only else None
    records = []
    for subset_name in subset_names:
        subset_samples = subsets[subset_name]
        print(f"\n>>> Running subset: {subset_name} ({len(subset_samples)} samples)")
        rows, summary = run_agent_eval(
            subset_samples,
            agent_runner=agent_runner,
            contract_reviewer=contract_reviewer,
            show_progress=True,
        )
        csv_path, json_path = save_results(
            rows,
            summary,
            results_dir=output_dir,
            tag=combined_tag(tag, subset_name),
        )
        print_summary(summary)
        print(f">>> CSV saved: {csv_path}")
        print(f">>> JSON saved: {json_path}")
        records.append(
            build_suite_record(subset_name, rows, summary, csv_path, json_path)
        )

    suite_path = write_suite_summary(
        records,
        testset_path=testset_path,
        output_dir=output_dir,
        suite=suite,
        tag=tag,
    )
    print_suite_table(records)
    print(f"\n>>> Suite summary saved: {suite_path}")
    return suite_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent Eval subsets and full suites.")
    parser.add_argument(
        "--testset",
        type=Path,
        default=AGENT_TESTSET_PATH,
        help="Agent eval testset path",
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default="all",
        help="Which subset suite to run",
    )
    parser.add_argument("--tag", type=str, default=None, help="Result tag prefix")
    parser.add_argument(
        "--include-contract-evidence",
        action="store_true",
        help="合同审查样本也检索法律依据；默认关闭以保持 Agent Eval 快速。",
    )
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="只评估本地路由、工具选择、拒答和合同风险；跳过 RAG/LLM 生成链路。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AGENT_RESULTS_DIR,
        help="CSV/JSON output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_suite(
        testset_path=args.testset,
        output_dir=args.output_dir,
        suite=args.suite,
        tag=args.tag,
        include_contract_evidence=args.include_contract_evidence,
        routing_only=args.routing_only,
    )


if __name__ == "__main__":
    main()
