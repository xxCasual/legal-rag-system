"""
run_evaluation.py
==================
对一个 RAG 策略跑完整评估，结果保存到 data/eval/results/<strategy>_<timestamp>.csv

用法：
    python run_evaluation.py --strategy baseline
    python run_evaluation.py --strategy hybrid
    python run_evaluation.py --strategy hybrid_rerank

可选参数：
    --limit N        只跑前 N 个样本（调试用）
    --tag XXX        给本次结果加自定义标签（如 baseline_chunk500）
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

import config
from rag_adapter import get_rag_system, STRATEGIES
from diagnostics import METRICS, read_replay_csv, score_samples, summarize


# ============================================================
# Step 1: 加载测试集
# ============================================================

def load_testset() -> List[Dict[str, Any]]:
    if not config.TESTSET_PATH.exists():
        print(f"[错误] 测试集不存在: {config.TESTSET_PATH}")
        print("       请先运行: python generate_testset.py")
        sys.exit(1)

    samples = json.loads(config.TESTSET_PATH.read_text(encoding="utf-8"))
    print(f">>> 加载测试集: {len(samples)} 个样本")
    return samples


# ============================================================
# Step 2: 用 RAG 系统跑测试集（生成 answer 和 contexts）
# ============================================================

def run_rag_on_testset(rag_system, samples: List[Dict]) -> List[Dict]:
    """
    对每个测试样本调用 RAG 系统，得到 (answer, contexts)。
    这一步是真正"花钱+花时间"的部分（取决于你 RAG 系统本身）。
    """
    print(f"\n>>> 用 [{rag_system.name}] 策略跑 {len(samples)} 个样本...")

    enriched = []
    failed = 0
    for sample in tqdm(samples, desc="Running RAG"):
        question = sample["question"]
        try:
            answer, contexts = rag_system.query(question)
            enriched.append({
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": sample["ground_truth"],
                "synthesizer_name": sample.get("synthesizer_name", "unknown"),
                "generation_status": "ok",
            })
        except Exception as e:
            print(f"\n[!] 样本失败: {question[:50]}... ({type(e).__name__})")
            failed += 1
            # 失败样本占位，避免整体崩溃
            enriched.append({
                "user_input": question,
                "response": "",
                "retrieved_contexts": [""],
                "reference": sample["ground_truth"],
                "synthesizer_name": sample.get("synthesizer_name", "unknown"),
                "generation_status": "error",
                "generation_error_type": type(e).__name__,
            })

    if failed:
        print(f"\n[!] {failed} 个样本失败，已用空答案占位")
    return enriched


# ============================================================
# Step 3: 用 RAGAS 评估四个指标
# ============================================================

def evaluate_with_ragas(enriched_samples: List[Dict], *, metrics=None, attempts=1, audit_path=None) -> pd.DataFrame:
    """Score one sample/metric at a time; retain all failures in a JSONL ledger."""
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    selected = list(dict.fromkeys(metrics or METRICS))
    unknown = set(selected) - set(METRICS)
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    llm = None
    embeddings = None
    factories = {"faithfulness": Faithfulness, "answer_relevancy": ResponseRelevancy,
                 "llm_context_precision_with_reference": LLMContextPrecisionWithReference, "context_recall": LLMContextRecall}
    instances = {}

    def scorer(sample, metric, attempt):
        nonlocal llm, embeddings
        # Initialization is inside the audited boundary, including embedding failures.
        if llm is None:
            llm = LangchainLLMWrapper(config.get_llm(model=config.EVALUATOR_LLM_MODEL, max_retries=0))
        if metric == "answer_relevancy" and embeddings is None:
            embeddings = LangchainEmbeddingsWrapper(config.get_embeddings())
        if metric not in instances:
            instances[metric] = factories[metric](llm=llm, **({"embeddings": embeddings} if metric == "answer_relevancy" else {}))
        data = {k: sample[k] for k in ("user_input", "response", "retrieved_contexts", "reference")}
        result = evaluate(dataset=EvaluationDataset.from_list([data]), metrics=[instances[metric]],
                          llm=llm, embeddings=embeddings, run_config=RunConfig(max_workers=1, timeout=180, max_retries=1),
                          show_progress=False, raise_exceptions=True)
        return result.to_pandas().iloc[0][instances[metric].name]

    audit_path = audit_path or config.RESULTS_DIR / ("attempts_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
    return pd.DataFrame(score_samples(enriched_samples, selected, scorer, attempts=attempts, audit_path=audit_path))


# ============================================================
# Step 4: 保存结果
# ============================================================

def save_results(df: pd.DataFrame, strategy_name: str, tag: str = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_{tag}" if tag else ""
    filename = f"{strategy_name}{suffix}_{timestamp}.csv"
    path = config.RESULTS_DIR / filename

    df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel 友好
    print(f"\n>>> 结果已保存: {path}")
    return path


def print_summary(df: pd.DataFrame, strategy_name: str):
    """Always display denominator and missingness; no aggregate of unlike denominators."""
    print(f"\n评估总结 - {strategy_name}")
    print("指标 | 有效/总数 | 缺失率 | 均值 | 标准差（题间，非重复稳定性）")
    metrics = [name for name in METRICS if name in df.columns]
    for metric, stat in summarize(df.to_dict("records"), metrics).items():
        mean = f"{stat['mean']:.4f}" if stat['mean'] is not None else "N/A"
        std = f"{stat['sample_std']:.4f}" if stat['sample_std'] is not None else "N/A"
        missing = f"{stat['missing_rate']:.1%}" if stat['missing_rate'] is not None else "N/A"
        print(f"{metric} | {stat['valid']}/{stat['total']} | {missing} | {mean} | {std}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default="baseline",
        help="要评估的 RAG 策略（回放时不生成新答案）",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只跑前 N 个样本（调试用）",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="给结果文件加自定义标签",
    )
    parser.add_argument("--replay-csv", type=Path, help="复用已保存的回答和上下文，仅重新评分")
    parser.add_argument("--out-dir", type=Path, help="独立结果目录，不覆盖历史 CSV")
    parser.add_argument("--metric", choices=METRICS, action="append", help="只评分指定指标，可重复")
    parser.add_argument("--attempts", type=int, default=1, help="每项最大尝试次数，默认不额外重试")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0: parser.error("--limit must be positive")
    if args.attempts <= 0: parser.error("--attempts must be positive")
    if args.out_dir:
        config.RESULTS_DIR = args.out_dir
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not config.DEEPSEEK_API_KEY:
        print("[错误] 未设置 DEEPSEEK_API_KEY")
        sys.exit(1)

    if args.replay_csv:
        enriched = read_replay_csv(args.replay_csv)
        if args.limit: enriched = enriched[:args.limit]
    else:
        samples = load_testset()
        if args.limit: samples = samples[:args.limit]
        rag = get_rag_system(args.strategy)
        enriched = run_rag_on_testset(rag, samples)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    audit = config.RESULTS_DIR / f"attempts_{stamp}.jsonl"
    # Preserve inputs even if model/embedding setup fails.
    (config.RESULTS_DIR / f"inputs_{stamp}.json").write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    import hashlib
    import importlib.metadata
    import platform
    meta = {"python": platform.python_version(), "strategy": args.strategy, "replay": str(args.replay_csv) if args.replay_csv else None,
            "model": config.EVALUATOR_LLM_MODEL, "metrics": args.metric or list(METRICS), "attempts": args.attempts,
            "packages": {name: importlib.metadata.version(name) for name in ("ragas", "langchain-openai", "pandas")},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "replay_sha256": hashlib.sha256(args.replay_csv.read_bytes()).hexdigest() if args.replay_csv else None,
            "audit": str(audit), "mode": "saved_answer_rescore" if args.replay_csv else "fresh_generation"}
    (config.RESULTS_DIR / f"manifest_{stamp}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    df = evaluate_with_ragas(enriched, metrics=args.metric, attempts=args.attempts, audit_path=audit)
    (config.RESULTS_DIR / f"summary_{stamp}.json").write_text(json.dumps(summarize(df.to_dict("records"), args.metric or METRICS), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    # 5. 保存 + 总结
    path = save_results(df, args.strategy, tag=args.tag)
    print_summary(df, args.strategy)

    print("\n>>> 下一步：")
    print(f"    1. 跑其他策略对比: python run_evaluation.py --strategy hybrid")
    print(f"    2. 查看可视化对比: python visualize_results.py")


if __name__ == "__main__":
    main()
