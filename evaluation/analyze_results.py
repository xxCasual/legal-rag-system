"""
evaluation/analyze_results.py
==============================
分析评估结果 CSV，定位"崩盘样本"——找出哪些样本拉低了均值。
跑完 baseline 评估后，立刻跑这个脚本看瓶颈在哪。

用法：
    python evaluation/analyze_results.py
    python evaluation/analyze_results.py --csv data/eval/results/baseline_xxx.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "eval" / "results"


METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "llm_context_precision_with_reference": "Context Precision",
    "context_precision": "Context Precision",  # 兼容旧版列名
    "context_recall": "Context Recall",
}


def find_latest_csv():
    csvs = sorted(RESULTS_DIR.glob("baseline_*.csv"))
    if not csvs:
        print(f"❌ 未找到 baseline csv: {RESULTS_DIR}")
        sys.exit(1)
    return csvs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="CSV 路径（默认: 最新的 baseline_*.csv）")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="低分阈值，低于此值视为崩盘（默认 0.5）")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else find_latest_csv()
    print(f">>> 分析: {csv_path.name}\n")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"总样本数: {len(df)}\n")

    # 找到实际存在的指标列
    actual_metrics = {col: name for col, name in METRICS.items() if col in df.columns}

    # ============================================================
    # 1. 整体分数分布
    # ============================================================
    print("=" * 70)
    print("【1】各指标分数分布")
    print("=" * 70)
    print(f"{'指标':<25}{'均值':>10}{'中位数':>10}{'<0.3':>8}{'<0.5':>8}{'=1.0':>8}")
    print("-" * 70)
    for col, name in actual_metrics.items():
        vals = df[col].dropna().astype(float)
        if len(vals) == 0:
            continue
        n_low = (vals < 0.3).sum()
        n_mid = (vals < 0.5).sum()
        n_perfect = (vals >= 0.99).sum()
        print(f"{name:<25}{vals.mean():>10.4f}{vals.median():>10.4f}"
              f"{n_low:>8}{n_mid:>8}{n_perfect:>8}")

    # ============================================================
    # 2. 找出"全方位崩盘"的样本（多个指标同时低）
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"【2】崩盘样本（≥2 个指标 < {args.threshold}）")
    print("=" * 70)

    metric_cols = list(actual_metrics.keys())
    df["low_count"] = df[metric_cols].apply(
        lambda row: sum(1 for v in row if pd.notna(v) and float(v) < args.threshold),
        axis=1
    )
    crashed = df[df["low_count"] >= 2].sort_values("low_count", ascending=False)

    print(f"\n崩盘样本数: {len(crashed)} / {len(df)} ({len(crashed)/len(df)*100:.1f}%)\n")

    if len(crashed) > 0:
        for i, (_, row) in enumerate(crashed.head(10).iterrows(), 1):
            print(f"\n──── 崩盘样本 {i}/{min(10, len(crashed))} ────")
            q = row.get("user_input", row.get("question", "?"))
            print(f"问题: {q}")

            # 类型（如果有）
            if "synthesizer_name" in row:
                print(f"类型: {row['synthesizer_name']}")

            # 打印各项分数
            print("分数:", end=" ")
            for col, name in actual_metrics.items():
                v = row.get(col)
                if pd.notna(v):
                    mark = "❌" if float(v) < args.threshold else "✓"
                    print(f"{name}={float(v):.2f}{mark}", end="  ")
            print()

            # 答案预览
            ans = row.get("response", row.get("answer", ""))
            if isinstance(ans, str) and ans:
                preview = ans[:150].replace("\n", " ")
                print(f"答案: {preview}{'...' if len(ans) > 150 else ''}")

            # ground truth
            gt = row.get("reference", row.get("ground_truth", ""))
            if isinstance(gt, str) and gt:
                preview = gt[:150].replace("\n", " ")
                print(f"标答: {preview}{'...' if len(gt) > 150 else ''}")

    # ============================================================
    # 3. 按问题类型分组（如果有）
    # ============================================================
    if "synthesizer_name" in df.columns:
        print(f"\n{'=' * 70}")
        print("【3】按问题类型分组的均值")
        print("=" * 70)
        cols_to_show = [c for c in metric_cols if c in df.columns]
        grouped = df.groupby("synthesizer_name")[cols_to_show].agg(["mean", "count"])
        print(grouped.round(3).to_string())

    # ============================================================
    # 4. 给出建议
    # ============================================================
    print(f"\n{'=' * 70}")
    print("【4】诊断建议")
    print("=" * 70)

    suggestions = []

    # 看哪个指标的低分样本最多
    low_counts = {}
    for col in metric_cols:
        vals = df[col].dropna().astype(float)
        if len(vals) > 0:
            low_counts[col] = (vals < args.threshold).sum()

    if low_counts:
        worst_col = max(low_counts, key=low_counts.get)
        worst_name = actual_metrics[worst_col]
        worst_n = low_counts[worst_col]

        print(f"\n👉 最薄弱指标: {worst_name} (低分样本 {worst_n}/{len(df)})")

        if "context_precision" in worst_col or "precision" in worst_col.lower():
            suggestions.append(
                "Precision 低 → 检索召回了无关 chunk。优先做 reranker 重排序，"
                "把无关 chunk 排到后面或截断"
            )
        if "context_recall" in worst_col or "recall" in worst_col.lower():
            suggestions.append(
                "Recall 低 → 关键 chunk 没召回。优先做 hybrid search（BM25+向量），"
                "BM25 对法条编号、专业术语更敏感"
            )
        if "faithfulness" in worst_col:
            suggestions.append(
                "Faithfulness 低 → 模型在编内容。检查 prompt 约束是否够强，"
                "或检查 context 是否被 CRAG 过度过滤导致信息不足"
            )
        if "answer_relevancy" in worst_col:
            suggestions.append(
                "Answer Relevancy 低 → 答案跑题。检查路由器准确率，"
                "或检查 prompt 模板是否让模型偏离了原问题"
            )

    # 中位数 vs 均值差距大
    for col, name in actual_metrics.items():
        vals = df[col].dropna().astype(float)
        if len(vals) > 0 and vals.median() - vals.mean() > 0.2:
            suggestions.append(
                f"{name}: 中位数({vals.median():.2f}) 远高于均值({vals.mean():.2f})，"
                f"说明少数样本严重崩盘。优先看上面"
                f"【崩盘样本】定位是测试集质量问题、路由错误，还是检索失败"
            )
            break

    if not suggestions:
        suggestions.append("各指标分布尚均匀，可考虑全面优化（hybrid + reranker）")

    for i, s in enumerate(suggestions, 1):
        print(f"\n  {i}. {s}")

    print()


if __name__ == "__main__":
    main()
