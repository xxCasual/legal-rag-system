"""
Agent workflow evaluation.

This script is intentionally separate from the RAGAS evaluation pipeline. It
checks discrete Agent behaviors such as intent routing, tool selection, refusal
handling, and contract-review risk classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENT_TESTSET_PATH = ROOT / "data" / "eval" / "agent_testset.json"
AGENT_RESULTS_DIR = ROOT / "data" / "eval" / "agent_results"
REQUIRED_FIELDS = (
    "query",
    "expected_intent",
    "expected_tools",
    "should_refuse",
    "expected_risk_level",
)
METRIC_COLUMNS = (
    "intent_match",
    "tool_call_match",
    "refusal_match",
    "risk_match",
)


AgentRunner = Callable[[str], Dict[str, Any]]
ContractReviewer = Callable[[str], Dict[str, Any]]


def load_agent_testset(path: Path | str = AGENT_TESTSET_PATH) -> List[Dict[str, Any]]:
    testset_path = Path(path)
    if not testset_path.exists():
        raise FileNotFoundError(f"Agent testset not found: {testset_path}")
    samples = json.loads(testset_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("Agent testset must be a JSON list")
    for index, sample in enumerate(samples):
        _validate_sample(sample, index)
    return samples


def run_agent_eval(
    samples: Sequence[Dict[str, Any]],
    agent_runner: AgentRunner | None = None,
    contract_reviewer: ContractReviewer | None = None,
    show_progress: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    agent_runner = agent_runner or _default_agent_runner
    contract_reviewer = contract_reviewer or _default_contract_reviewer

    rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        _validate_sample(sample, index)
        if show_progress:
            task_type = (
                "contract_review"
                if sample["expected_risk_level"] is not None
                else "agent_chat"
            )
            print(f">>> Agent Eval sample {index + 1}/{len(samples)} [{task_type}]")
        row = _evaluate_sample(sample, index, agent_runner, contract_reviewer)
        if show_progress:
            status = "failed" if row.get("error") else "ok"
            print(f"    {status} in {row['latency']}s")
        rows.append(row)

    summary = build_summary(rows)
    return rows, summary


def build_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics = {
        "intent_accuracy": _metric_summary(rows, "intent_match"),
        "tool_call_accuracy": _metric_summary(rows, "tool_call_match"),
        "refusal_accuracy": _metric_summary(rows, "refusal_match"),
        "risk_accuracy": _metric_summary(rows, "risk_match"),
    }
    return {
        "total_samples": len(rows),
        "failed_samples": sum(1 for row in rows if row.get("error")),
        "metrics": metrics,
    }


def save_results(
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    results_dir: Path | str = AGENT_RESULTS_DIR,
    tag: str | None = None,
) -> Tuple[Path, Path]:
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    csv_path = output_dir / f"agent_eval{suffix}_{timestamp}.csv"
    json_path = output_dir / f"agent_eval{suffix}_{timestamp}.json"

    fieldnames = [
        "sample_id",
        "task_type",
        "query",
        "expected_intent",
        "actual_intent",
        "expected_tools",
        "actual_tools",
        "expected_risk_level",
        "actual_risk_level",
        "should_refuse",
        "actual_refused",
        "intent_match",
        "tool_call_match",
        "refusal_match",
        "risk_match",
        "latency",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})

    summary_with_path = {
        **summary,
        "csv_path": str(csv_path),
    }
    json_path.write_text(
        json.dumps(summary_with_path, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("Agent Eval Summary")
    print("=" * 70)
    print(f"total_samples: {summary['total_samples']}")
    print(f"failed_samples: {summary['failed_samples']}")
    for metric_name, metric in summary["metrics"].items():
        score = metric["score"]
        score_text = "n/a" if score is None else f"{score:.4f}"
        print(
            f"{metric_name}: {score_text} "
            f"({metric['correct']}/{metric['valid_samples']})"
        )
    print("=" * 70)


def _evaluate_sample(
    sample: Dict[str, Any],
    index: int,
    agent_runner: AgentRunner,
    contract_reviewer: ContractReviewer,
) -> Dict[str, Any]:
    query = sample["query"]
    expected_intent = sample["expected_intent"]
    expected_tools = sample["expected_tools"]
    expected_risk_level = sample["expected_risk_level"]
    should_refuse = sample["should_refuse"]
    is_contract_review = expected_risk_level is not None

    row = {
        "sample_id": index,
        "task_type": "contract_review" if is_contract_review else "agent_chat",
        "query": query,
        "expected_intent": expected_intent,
        "actual_intent": None,
        "expected_tools": expected_tools,
        "actual_tools": [],
        "expected_risk_level": expected_risk_level,
        "actual_risk_level": None,
        "should_refuse": should_refuse,
        "actual_refused": False,
        "intent_match": None,
        "tool_call_match": None,
        "refusal_match": None,
        "risk_match": None,
        "latency": 0,
        "error": "",
    }

    started_at = time.perf_counter()
    try:
        if is_contract_review:
            result = contract_reviewer(query)
            row["actual_tools"] = ["contract_review_rules"]
            row["actual_risk_level"] = result.get("risk_level")
            row["actual_refused"] = result.get("risk_level") == "high"
        else:
            result = agent_runner(query)
            row["actual_intent"] = result.get("intent")
            row["actual_tools"] = result.get("tools_used", [])
            row["actual_refused"] = _looks_like_refusal(result.get("answer", ""))
    except Exception as exc:
        row["error"] = str(exc)
    finally:
        row["latency"] = round(time.perf_counter() - started_at, 3)

    failed = bool(row["error"])
    row["intent_match"] = (
        None
        if expected_intent is None
        else False
        if failed
        else row["actual_intent"] == expected_intent
    )
    row["tool_call_match"] = False if failed else _tools_match(expected_tools, row["actual_tools"])
    row["refusal_match"] = False if failed else row["actual_refused"] == should_refuse
    row["risk_match"] = (
        None
        if expected_risk_level is None
        else False
        if failed
        else row["actual_risk_level"] == expected_risk_level
    )
    return row


def _metric_summary(rows: Sequence[Dict[str, Any]], column: str) -> Dict[str, Any]:
    valid_values = [row[column] for row in rows if row.get(column) is not None]
    correct = sum(1 for value in valid_values if value is True)
    total = len(valid_values)
    return {
        "score": round(correct / total, 4) if total else None,
        "correct": correct,
        "valid_samples": total,
    }


def _tools_match(expected_tools: Sequence[str], actual_tools: Sequence[str]) -> bool:
    if not expected_tools:
        return not actual_tools
    return set(expected_tools).issubset(set(actual_tools))


def _looks_like_refusal(answer: str) -> bool:
    refusal_terms = (
        "无法回答",
        "无法提供",
        "不能回答",
        "知识库外",
        "未检索到",
        "不予输出",
        "人工复核",
        "pending_review",
        "拒绝",
    )
    return any(term in answer for term in refusal_terms)


def _validate_sample(sample: Dict[str, Any], index: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in sample]
    if missing:
        raise ValueError(f"Sample #{index} missing required fields: {missing}")
    if not isinstance(sample["query"], str) or not sample["query"].strip():
        raise ValueError(f"Sample #{index} query must be a non-empty string")
    if sample["expected_intent"] not in {None, "law_qa", "policy_qa"}:
        raise ValueError(f"Sample #{index} expected_intent is invalid")
    if not isinstance(sample["expected_tools"], list):
        raise ValueError(f"Sample #{index} expected_tools must be a list")
    if not isinstance(sample["should_refuse"], bool):
        raise ValueError(f"Sample #{index} should_refuse must be a boolean")
    if sample["expected_risk_level"] not in {None, "low", "medium", "high"}:
        raise ValueError(f"Sample #{index} expected_risk_level is invalid")


def _default_agent_runner(query: str) -> Dict[str, Any]:
    from app.agent import run_agent_chat

    return run_agent_chat(query)


def _default_contract_reviewer(contract_text: str) -> Dict[str, Any]:
    from app.services.contract_review_service import contract_review_service

    return contract_review_service.review_contract(contract_text, include_evidence=False)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testset",
        type=Path,
        default=AGENT_TESTSET_PATH,
        help="Agent eval testset path",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 个样本")
    parser.add_argument("--tag", type=str, default=None, help="给结果文件加自定义标签")
    parser.add_argument(
        "--include-contract-evidence",
        action="store_true",
        help="合同审查样本也检索法律依据；默认关闭以保持 Agent Eval 快速。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AGENT_RESULTS_DIR,
        help="CSV/JSON output directory",
    )
    args = parser.parse_args()

    samples = load_agent_testset(args.testset)
    if args.limit:
        samples = samples[: args.limit]
        print(f">>> 限制为前 {args.limit} 个样本")

    contract_reviewer = None
    if args.include_contract_evidence:
        from app.services.contract_review_service import contract_review_service

        contract_reviewer = (
            lambda text: contract_review_service.review_contract(
                text,
                include_evidence=True,
            )
        )

    rows, summary = run_agent_eval(
        samples,
        contract_reviewer=contract_reviewer,
        show_progress=True,
    )
    csv_path, json_path = save_results(rows, summary, args.output_dir, args.tag)
    print_summary(summary)
    print(f"\n>>> CSV saved: {csv_path}")
    print(f">>> JSON saved: {json_path}")


if __name__ == "__main__":
    main()
