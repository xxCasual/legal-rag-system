"""
run_evaluation.py
==================
对一个 RAG 策略跑完整评估，结果保存到 data/results/<strategy>_<timestamp>.csv

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
from datasets import Dataset
from tqdm import tqdm

import config
from rag_adapter import get_rag_system, STRATEGIES


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
            })
        except Exception as e:
            print(f"\n[!] 样本失败: {question[:50]}... ({e})")
            failed += 1
            # 失败样本占位，避免整体崩溃
            enriched.append({
                "user_input": question,
                "response": "",
                "retrieved_contexts": [""],
                "reference": sample["ground_truth"],
                "synthesizer_name": sample.get("synthesizer_name", "unknown"),
            })

    if failed:
        print(f"\n[!] {failed} 个样本失败，已用空答案占位")
    return enriched


# ============================================================
# Step 3: 用 RAGAS 评估四个指标
# ============================================================

def evaluate_with_ragas(enriched_samples: List[Dict]) -> pd.DataFrame:
    """
    跑 RAGAS 评估。返回每个样本的指标分数 + 元数据。
    """
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import (
        Faithfulness,
        ResponseRelevancy,        # 旧名 AnswerRelevancy
        LLMContextPrecisionWithReference,  # 旧名 ContextPrecision
        LLMContextRecall,          # 旧名 ContextRecall
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    print(f"\n>>> 配置 RAGAS 评估器...")
    print(f"    LLM: {config.EVALUATOR_LLM_MODEL} (DeepSeek)")

    evaluator_llm = LangchainLLMWrapper(
        config.get_llm(model=config.EVALUATOR_LLM_MODEL)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        config.get_embeddings()
    )

    # 构造 RAGAS 数据集
    eval_dataset = EvaluationDataset.from_list(enriched_samples)

    # 四个核心指标
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings
        ),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    print(f">>> 开始评估（4 个指标 × {len(enriched_samples)} 样本）...")
    print(f"    预计耗时 10-25 分钟")

    run_config = RunConfig(
        max_workers=config.MAX_CONCURRENT,
        timeout=180,
        max_retries=3,
    )

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        show_progress=True,
    )

    df = result.to_pandas()
    return df


# ============================================================
# Step 4: 保存结果
# ============================================================

def save_results(df: pd.DataFrame, strategy_name: str, tag: str = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    filename = f"{strategy_name}{suffix}_{timestamp}.csv"
    path = config.RESULTS_DIR / filename

    df.to_csv(path, index=False, encoding="utf-8-sig")  # utf-8-sig: Excel 友好
    print(f"\n>>> 结果已保存: {path}")
    return path


def print_summary(df: pd.DataFrame, strategy_name: str):
    """打印四个指标的均值，作为本次评估的"基线分数"。"""
    print("\n" + "=" * 70)
    print(f"评估总结 - 策略: {strategy_name}")
    print("=" * 70)

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall",
    ]
    # RAGAS 列名跨版本可能略有不同，做兜底
    name_map = {
        "faithfulness": "Faithfulness (无幻觉)",
        "answer_relevancy": "Answer Relevancy (切题)",
        "llm_context_precision_with_reference": "Context Precision (检索精准度)",
        "context_precision": "Context Precision (检索精准度)",
        "context_recall": "Context Recall (检索召回)",
    }

    print(f"\n{'指标':<35} {'均值':>10} {'中位数':>10} {'标准差':>10}")
    print("-" * 70)
    for col in df.columns:
        if col in name_map:
            try:
                vals = df[col].dropna().astype(float)
                if len(vals) == 0:
                    continue
                print(
                    f"{name_map[col]:<35} "
                    f"{vals.mean():>10.4f} "
                    f"{vals.median():>10.4f} "
                    f"{vals.std():>10.4f}"
                )
            except Exception:
                pass

    # 按问题类型分组的均值（如果有 synthesizer_name 列）
    if "synthesizer_name" in df.columns:
        print(f"\n按问题类型分组的均值:")
        print("-" * 70)
        metric_cols_present = [
            c for c in metric_cols if c in df.columns
        ]
        if metric_cols_present:
            grouped = df.groupby("synthesizer_name")[metric_cols_present].mean()
            print(grouped.round(4).to_string())

    print("=" * 70)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        required=True,
        help="要评估的 RAG 策略",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只跑前 N 个样本（调试用）",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="给结果文件加自定义标签",
    )
    args = parser.parse_args()

    if not config.DEEPSEEK_API_KEY:
        print("[错误] 未设置 DEEPSEEK_API_KEY")
        sys.exit(1)

    # 1. 加载测试集
    samples = load_testset()
    if args.limit:
        samples = samples[: args.limit]
        print(f">>> 限制为前 {args.limit} 个样本（调试模式）")

    # 2. 实例化 RAG 系统
    print(f"\n>>> 实例化策略: {args.strategy}")
    rag = get_rag_system(args.strategy)

    # 3. 跑 RAG（生成 answer + contexts）
    enriched = run_rag_on_testset(rag, samples)

    # 4. RAGAS 评估
    df = evaluate_with_ragas(enriched)

    # 5. 保存 + 总结
    path = save_results(df, args.strategy, tag=args.tag)
    print_summary(df, args.strategy)

    print("\n>>> 下一步：")
    print(f"    1. 跑其他策略对比: python run_evaluation.py --strategy hybrid")
    print(f"    2. 查看可视化对比: python visualize_results.py")


if __name__ == "__main__":
    main()
