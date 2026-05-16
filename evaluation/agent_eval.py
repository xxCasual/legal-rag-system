"""
Agent workflow evaluation (optimized).

This script evaluates discrete Agent behaviors such as:
- intent routing
- tool selection
- refusal handling
- contract-review risk classification

Compared with the original version, this optimized version adds:
- suite filtering: all / law_qa / policy_qa / contract_review / refusal
- optional warmup to avoid counting model/reranker first-load time
- optional concurrent execution with --max-workers
- strict/subset tool matching
- raw_result recording for debugging
- cache support for repeated local debugging
- latency summary: avg / p95 / max / slowest samples
- two eval modes:
  - component: contract_review samples call contract_review_service directly
  - e2e: every sample goes through the Agent runner
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENT_TESTSET_PATH = ROOT / "data" / "eval" / "agent_testset.json"
AGENT_RESULTS_DIR = ROOT / "data" / "eval" / "agent_results"
AGENT_CACHE_PATH = AGENT_RESULTS_DIR / "agent_eval_cache.json"

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

ALLOWED_SUITES = ("all", "law_qa", "policy_qa", "contract_review", "refusal")
ALLOWED_EVAL_MODES = ("component", "e2e")
ALLOWED_TOOL_MATCH_MODES = ("strict", "subset")


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


def filter_samples_by_suite(
    samples: Sequence[Dict[str, Any]],
    suite: str,
) -> List[Dict[str, Any]]:
    if suite == "all":
        return list(samples)

    if suite == "law_qa":
        return [
            sample for sample in samples
            if sample["expected_intent"] == "law_qa"
        ]

    if suite == "policy_qa":
        return [
            sample for sample in samples
            if sample["expected_intent"] == "policy_qa"
        ]

    if suite == "contract_review":
        return [
            sample for sample in samples
            if sample["expected_risk_level"] is not None
        ]

    if suite == "refusal":
        return [
            sample for sample in samples
            if sample["should_refuse"] is True
        ]

    raise ValueError(f"Unknown suite: {suite}")


def run_agent_eval(
    samples: Sequence[Dict[str, Any]],
    agent_runner: AgentRunner | None = None,
    contract_reviewer: ContractReviewer | None = None,
    show_progress: bool = False,
    max_workers: int = 1,
    tool_match_mode: str = "strict",
    eval_mode: str = "e2e",
    cache: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if eval_mode not in ALLOWED_EVAL_MODES:
        raise ValueError(f"eval_mode must be one of {ALLOWED_EVAL_MODES}")

    if tool_match_mode not in ALLOWED_TOOL_MATCH_MODES:
        raise ValueError(f"tool_match_mode must be one of {ALLOWED_TOOL_MATCH_MODES}")

    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")

    agent_runner = agent_runner or _default_agent_runner
    contract_reviewer = contract_reviewer or _default_contract_reviewer
    cache_lock = threading.Lock()

    if max_workers == 1:
        rows: List[Dict[str, Any]] = []

        for index, sample in enumerate(samples):
            _validate_sample(sample, index)
            task_type = _task_type_for_sample(sample, eval_mode)

            if show_progress:
                print(f">>> Agent Eval sample {index + 1}/{len(samples)} [{task_type}]")

            row = _evaluate_sample(
                sample=sample,
                index=index,
                agent_runner=agent_runner,
                contract_reviewer=contract_reviewer,
                eval_mode=eval_mode,
                tool_match_mode=tool_match_mode,
                cache=cache,
                cache_lock=cache_lock,
            )

            if show_progress:
                status = "failed" if row.get("error") else "ok"
                cache_text = " cache" if row.get("cache_hit") else ""
                print(f"    {status}{cache_text} in {row['latency']}s")

            rows.append(row)

        summary = build_summary(rows)
        return rows, summary

    rows_with_slots: List[Dict[str, Any] | None] = [None] * len(samples)

    def run_one(index: int, sample: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        _validate_sample(sample, index)
        row = _evaluate_sample(
            sample=sample,
            index=index,
            agent_runner=agent_runner,
            contract_reviewer=contract_reviewer,
            eval_mode=eval_mode,
            tool_match_mode=tool_match_mode,
            cache=cache,
            cache_lock=cache_lock,
        )
        return index, row

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_one, index, sample)
            for index, sample in enumerate(samples)
        ]

        for future in as_completed(futures):
            index, row = future.result()
            rows_with_slots[index] = row

            if show_progress:
                status = "failed" if row.get("error") else "ok"
                cache_text = " cache" if row.get("cache_hit") else ""
                print(
                    f">>> Agent Eval sample {index + 1}/{len(samples)} "
                    f"{status}{cache_text} in {row['latency']}s"
                )

    rows = [row for row in rows_with_slots if row is not None]
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
        "cache_hits": sum(1 for row in rows if row.get("cache_hit")),
        "metrics": metrics,
        "latency": _latency_summary(rows),
        "failed_sample_ids": [
            row["sample_id"] for row in rows if row.get("error")
        ],
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
        "eval_mode",
        "task_type",
        "query",
        "expected_intent",
        "actual_intent",
        "intent_source",
        "intent_confidence",
        "actual_route",
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
        "runner_latency",
        "latency",
        "cache_hit",
        "error",
        "raw_result",
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
    print(f"cache_hits: {summary.get('cache_hits', 0)}")

    print("\nMetrics")
    for metric_name, metric in summary["metrics"].items():
        score = metric["score"]
        score_text = "n/a" if score is None else f"{score:.4f}"
        print(
            f"{metric_name}: {score_text} "
            f"({metric['correct']}/{metric['valid_samples']})"
        )

    latency = summary.get("latency", {})
    print("\nLatency")
    print(f"avg: {latency.get('avg')}")
    print(f"p95: {latency.get('p95')}")
    print(f"max: {latency.get('max')}")

    slowest_samples = latency.get("slowest_samples") or []
    if slowest_samples:
        print("\nSlowest samples")
        for item in slowest_samples:
            query_preview = item["query"].replace("\n", " ")[:80]
            print(
                f"  #{item['sample_id']} "
                f"[{item['task_type']}] "
                f"{item['latency']}s - {query_preview}"
            )

    failed_sample_ids = summary.get("failed_sample_ids") or []
    if failed_sample_ids:
        print("\nFailed sample ids")
        print(", ".join(str(item) for item in failed_sample_ids))

    print("=" * 70)


def load_cache(path: Path | str = AGENT_CACHE_PATH) -> Dict[str, Any]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f">>> Cache file is invalid JSON, ignored: {cache_path}")
        return {}

    if not isinstance(data, dict):
        print(f">>> Cache file is not a JSON object, ignored: {cache_path}")
        return {}

    return data


def save_cache(cache: Dict[str, Any], path: Path | str = AGENT_CACHE_PATH) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def warmup_agent(agent_runner: AgentRunner | None = None) -> None:
    agent_runner = agent_runner or _default_agent_runner

    print(">>> Warming up agent...")
    started_at = time.perf_counter()

    try:
        agent_runner("请简单回答：你好")
    except Exception as exc:
        print(f">>> Agent warmup failed: {type(exc).__name__}: {exc}")
        return

    latency = round(time.perf_counter() - started_at, 3)
    print(f">>> Agent warmup finished in {latency}s")


def warmup_contract_reviewer(
    contract_reviewer: ContractReviewer | None = None,
) -> None:
    contract_reviewer = contract_reviewer or _default_contract_reviewer

    print(">>> Warming up contract reviewer...")
    started_at = time.perf_counter()

    try:
        contract_reviewer(
            "合同期限一年。试用期一个月。工资每月一万元。"
            "执行标准工时。公司依法缴纳社会保险。"
        )
    except Exception as exc:
        print(f">>> Contract reviewer warmup failed: {type(exc).__name__}: {exc}")
        return

    latency = round(time.perf_counter() - started_at, 3)
    print(f">>> Contract reviewer warmup finished in {latency}s")


def _evaluate_sample(
    sample: Dict[str, Any],
    index: int,
    agent_runner: AgentRunner,
    contract_reviewer: ContractReviewer,
    eval_mode: str,
    tool_match_mode: str,
    cache: Dict[str, Any] | None,
    cache_lock: threading.Lock,
) -> Dict[str, Any]:
    query = sample["query"]
    expected_intent = sample["expected_intent"]
    expected_tools = sample["expected_tools"]
    expected_risk_level = sample["expected_risk_level"]
    should_refuse = sample["should_refuse"]

    is_contract_review = expected_risk_level is not None
    use_component_contract = eval_mode == "component" and is_contract_review

    row: Dict[str, Any] = {
        "sample_id": index,
        "eval_mode": eval_mode,
        "task_type": _task_type_for_sample(sample, eval_mode),
        "query": query,
        "expected_intent": expected_intent,
        "actual_intent": None,
        "intent_source": "",
        "intent_confidence": None,
        "actual_route": None,
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
        "runner_latency": None,
        "latency": 0,
        "cache_hit": False,
        "error": "",
        "raw_result": {},
    }

    started_at = time.perf_counter()
    result: Dict[str, Any] = {}

    try:
        cache_key = _cache_key(sample, eval_mode)

        if cache is not None and cache_key in cache:
            cached = cache[cache_key]
            row["cache_hit"] = True
            result = cached.get("result", {}) or {}
            row["error"] = cached.get("error", "") or ""
            row["runner_latency"] = cached.get("runner_latency")
        else:
            call_started_at = time.perf_counter()

            if use_component_contract:
                result = contract_reviewer(query)
            else:
                result = agent_runner(query)

            runner_latency = _safe_float(result.get("latency"))
            if runner_latency is None:
                runner_latency = round(time.perf_counter() - call_started_at, 3)

            row["runner_latency"] = runner_latency

            if cache is not None:
                with cache_lock:
                    cache[cache_key] = {
                        "result": result,
                        "error": "",
                        "runner_latency": runner_latency,
                        "cached_at": datetime.now().isoformat(timespec="seconds"),
                    }

        if not isinstance(result, dict):
            raise TypeError(f"Runner result must be a dict, got {type(result).__name__}")

        row["raw_result"] = result

        if not row["error"]:
            if use_component_contract:
                # Component mode directly tests contract_review_service.
                # It does not prove that the Agent selected contract_review_rules.
                row["actual_tools"] = ["contract_review_rules"]
                row["actual_risk_level"] = _extract_risk_level(result)
                row["actual_refused"] = (
                    row["actual_risk_level"] == "high"
                    or _extract_refused(result)
                )
            else:
                row["actual_intent"] = _empty_to_none(result.get("intent"))
                row["intent_source"] = result.get("intent_source", "") or ""
                row["intent_confidence"] = _safe_float(result.get("intent_confidence"))
                row["actual_route"] = _empty_to_none(result.get("route"))
                row["actual_tools"] = _normalize_tools(result.get("tools_used", []))
                row["actual_risk_level"] = _extract_risk_level(result)
                row["actual_refused"] = _extract_refused(result)

    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"

        if cache is not None:
            cache_key = _cache_key(sample, eval_mode)
            with cache_lock:
                cache[cache_key] = {
                    "result": result if isinstance(result, dict) else {},
                    "error": row["error"],
                    "runner_latency": row.get("runner_latency"),
                    "cached_at": datetime.now().isoformat(timespec="seconds"),
                }

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

    row["tool_call_match"] = (
        None
        if use_component_contract
        else False
        if failed
        else _tools_match(
            expected_tools,
            row["actual_tools"],
            mode=tool_match_mode,
        )
    )

    row["refusal_match"] = (
        False
        if failed
        else row["actual_refused"] == should_refuse
    )

    row["risk_match"] = (
        None
        if expected_risk_level is None
        else False
        if failed
        else row["actual_risk_level"] == expected_risk_level
    )

    return row


def _task_type_for_sample(sample: Dict[str, Any], eval_mode: str) -> str:
    is_contract_review = sample["expected_risk_level"] is not None

    if eval_mode == "component" and is_contract_review:
        return "contract_review"

    return "agent_chat"


def _metric_summary(rows: Sequence[Dict[str, Any]], column: str) -> Dict[str, Any]:
    valid_values = [row[column] for row in rows if row.get(column) is not None]
    correct = sum(1 for value in valid_values if value is True)
    total = len(valid_values)

    return {
        "score": round(correct / total, 4) if total else None,
        "correct": correct,
        "valid_samples": total,
    }


def _latency_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("error")]
    latencies = [row["latency"] for row in valid_rows]

    if not latencies:
        return {
            "avg": None,
            "p95": None,
            "max": None,
            "slowest_samples": [],
        }

    sorted_latencies = sorted(latencies)
    p95_index = max(int(len(sorted_latencies) * 0.95) - 1, 0)

    slowest_samples = sorted(
        [
            {
                "sample_id": row["sample_id"],
                "task_type": row["task_type"],
                "latency": row["latency"],
                "query": row["query"],
            }
            for row in valid_rows
        ],
        key=lambda item: item["latency"],
        reverse=True,
    )[:5]

    return {
        "avg": round(sum(latencies) / len(latencies), 3),
        "p95": sorted_latencies[p95_index],
        "max": max(latencies),
        "slowest_samples": slowest_samples,
    }


def _tools_match(
    expected_tools: Sequence[str],
    actual_tools: Sequence[str],
    mode: str = "strict",
) -> bool:
    expected = set(expected_tools)
    actual = set(actual_tools)

    if mode == "strict":
        return expected == actual

    if mode == "subset":
        return expected.issubset(actual)

    raise ValueError(f"Unknown tool match mode: {mode}")


def _looks_like_refusal(answer: str) -> bool:
    refusal_terms = (
        "无法回答",
        "无法提供",
        "不能回答",
        "知识库外",
        "知识库中没有",
        "未检索到",
        "没有检索到",
        "未找到相关依据",
        "超出范围",
        "不予输出",
        "人工复核",
        "pending_review",
        "拒绝",
    )
    return any(term in answer for term in refusal_terms)


def _extract_refused(result: Dict[str, Any]) -> bool:
    refused = result.get("refused")

    if isinstance(refused, bool):
        return refused

    if isinstance(refused, str):
        normalized = refused.strip().lower()
        if normalized in {"true", "yes", "1", "refused"}:
            return True
        if normalized in {"false", "no", "0", "not_refused"}:
            return False

    status = result.get("status")
    if isinstance(status, str) and status.strip().lower() in {
        "refused",
        "rejected",
        "pending_review",
    }:
        return True

    review_status = result.get("review_status")
    if isinstance(review_status, str) and review_status.strip().lower() == "pending_review":
        return True

    for key in ("route", "result_type", "intent"):
        value = result.get(key)
        if isinstance(value, str) and value.strip().lower() == "refusal":
            return True

    answer = result.get("answer", "")
    if isinstance(answer, str):
        return _looks_like_refusal(answer)

    return False


def _extract_risk_level(result: Dict[str, Any]) -> str | None:
    candidates = (
        result.get("risk_level"),
        result.get("overall_risk"),
        result.get("risk"),
        result.get("level"),
    )

    for value in candidates:
        if not isinstance(value, str):
            continue

        normalized = value.strip().lower()
        if normalized in {"low", "medium", "high"}:
            return normalized

    return None


def _normalize_tools(tools: Any) -> List[str]:
    if tools is None:
        return []

    if isinstance(tools, str):
        return [tools] if tools else []

    if not isinstance(tools, Sequence):
        return [str(tools)]

    normalized: List[str] = []

    for tool in tools:
        if isinstance(tool, str):
            if tool:
                normalized.append(tool)
            continue

        if isinstance(tool, dict):
            name = (
                tool.get("name")
                or tool.get("tool")
                or tool.get("tool_name")
                or tool.get("function")
            )
            if name:
                normalized.append(str(name))
            continue

        normalized.append(str(tool))

    return normalized


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

    return contract_review_service.review_contract(
        contract_text,
        include_evidence=False,
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def _empty_to_none(value: Any) -> Any:
    if value == "":
        return None
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _cache_key(sample: Dict[str, Any], eval_mode: str) -> str:
    payload = {
        "eval_mode": eval_mode,
        "query": sample["query"],
        "expected_risk_level": sample.get("expected_risk_level"),
    }

    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--testset",
        type=Path,
        default=AGENT_TESTSET_PATH,
        help="Agent eval testset path",
    )

    parser.add_argument(
        "--suite",
        choices=ALLOWED_SUITES,
        default="all",
        help="选择要运行的评测子集",
    )

    parser.add_argument(
        "--eval-mode",
        choices=ALLOWED_EVAL_MODES,
        default="e2e",
        help=(
            "e2e: 所有样本都走 Agent；"
            "component: 合同审查样本直接调用合同审查组件"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑前 N 个样本。注意：会先按 suite 过滤，再 limit。",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="给结果文件加自定义标签",
    )

    parser.add_argument(
        "--include-contract-evidence",
        action="store_true",
        help="component 模式下，合同审查样本也检索法律依据；默认关闭以保持快速。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AGENT_RESULTS_DIR,
        help="CSV/JSON output directory",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并发样本数量。DeepSeek API + ChromaDB 建议先用 2 或 3。",
    )

    parser.add_argument(
        "--tool-match-mode",
        choices=ALLOWED_TOOL_MATCH_MODES,
        default="strict",
        help="strict 要求工具集合完全一致；subset 只要求包含期望工具。",
    )

    parser.add_argument(
        "--warmup",
        action="store_true",
        help="正式评测前预热 Agent / 合同审查器，避免首条样本包含模型加载时间。",
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="启用 JSON 缓存，适合反复调试同一批样本。",
    )

    parser.add_argument(
        "--cache-path",
        type=Path,
        default=AGENT_CACHE_PATH,
        help="缓存文件路径。",
    )

    args = parser.parse_args()

    samples = load_agent_testset(args.testset)
    samples = filter_samples_by_suite(samples, args.suite)

    if args.limit is not None:
        samples = samples[: args.limit]

    print("=" * 70)
    print(f"suite: {args.suite}")
    print(f"eval_mode: {args.eval_mode}")
    print(f"total_samples: {len(samples)}")
    print(f"max_workers: {args.max_workers}")
    print(f"tool_match_mode: {args.tool_match_mode}")
    print(f"cache: {args.cache}")
    print("=" * 70)

    contract_reviewer = None
    if args.include_contract_evidence:
        from app.services.contract_review_service import contract_review_service

        contract_reviewer = (
            lambda text: contract_review_service.review_contract(
                text,
                include_evidence=True,
            )
        )

    if args.warmup:
        if args.eval_mode == "e2e":
            warmup_agent()
        else:
            # component mode may use both paths when suite=all.
            if args.suite in {"all", "law_qa", "policy_qa", "refusal"}:
                warmup_agent()

            if args.suite in {"all", "contract_review"}:
                warmup_contract_reviewer(contract_reviewer)

    cache = load_cache(args.cache_path) if args.cache else None

    rows, summary = run_agent_eval(
        samples=samples,
        contract_reviewer=contract_reviewer,
        show_progress=True,
        max_workers=args.max_workers,
        tool_match_mode=args.tool_match_mode,
        eval_mode=args.eval_mode,
        cache=cache,
    )

    if args.cache and cache is not None:
        save_cache(cache, args.cache_path)
        print(f"\n>>> Cache saved: {args.cache_path}")

    csv_path, json_path = save_results(
        rows=rows,
        summary=summary,
        results_dir=args.output_dir,
        tag=args.tag,
    )

    print_summary(summary)

    print(f"\n>>> CSV saved: {csv_path}")
    print(f">>> JSON saved: {json_path}")


if __name__ == "__main__":
    main()
