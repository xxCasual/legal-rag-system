"""Historical metric coverage charts; no composite ranking or causal claims."""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import config
from diagnostics import summarize

# 中文字体
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "PingFang SC",
    "Arial Unicode MS", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================
# RAGAS 跨版本指标列名映射
# ============================================================

METRIC_ALIASES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "context_precision",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}

METRIC_DISPLAY = {
    "context_recall": "Context Recall\n(检索召回)",
    "context_precision": "Context Precision\n(检索精准度)",
    "faithfulness": "Faithfulness\n(无幻觉)",
    "answer_relevancy": "Answer Relevancy\n(切题)",
}

# 指标顺序（在所有图表中保持一致）
METRIC_ORDER = [
    "context_recall",
    "context_precision",
    "faithfulness",
    "answer_relevancy",
]


# ============================================================
# 演进顺序定义（项目主线）
# ============================================================

# 每个策略的"演进序号"和展示名称，决定柱状图、折线图的顺序和颜色
EVOLUTION_ORDER = {
    "baseline":             {"order": 0, "label": "v1\nbaseline",      "color": "#95A5A6", "kind": "stage"},
    "baseline_v2":          {"order": 1, "label": "v2\n修路由+清洗",    "color": "#5DADE2", "kind": "stage"},
    "baseline_v2_clean":    {"order": 1, "label": "v2\n修路由+清洗",    "color": "#5DADE2", "kind": "stage"},
    "hybrid":               {"order": 2, "label": "v3\nhybrid",        "color": "#3498DB", "kind": "stage"},
    "hybrid_v1":            {"order": 2, "label": "v3\nhybrid",        "color": "#3498DB", "kind": "stage"},
    "hybrid_v2":            {"order": 2, "label": "v3\nhybrid",        "color": "#3498DB", "kind": "stage"},
    "hybrid_rerank_v1":     {"order": 3, "label": "v4\n+rerank-base", "color": "#E74C3C", "kind": "stage"},
    "hybrid_rerank":        {"order": 3, "label": "v4\n+rerank-base", "color": "#E74C3C", "kind": "stage"},
    "hybrid_rerank_v2_m3":  {"order": 4, "label": "v5\n+rerank-m3",    "color": "#27AE60", "kind": "final"},
    "hybrid_rerank_v2":     {"order": 4, "label": "v5\n+rerank-m3",    "color": "#27AE60", "kind": "final"},
}

# 兜底颜色（未知策略名）
DEFAULT_COLOR = "#9B59B6"


# ============================================================
# 加载数据
# ============================================================

def parse_filename(path: Path) -> Dict[str, str]:
    """从 <strategy>[_<tag>]_<YYYYMMDD>_<HHMMSS>.csv 提取信息"""
    stem = path.stem
    m = re.match(r"^(.+?)_(\d{8})_(\d{6})(?:_(\d{6}))?$", stem)
    if not m:
        return {"strategy_full": stem, "timestamp": "unknown"}
    return {
        "strategy_full": m.group(1),
        "timestamp": f"{m.group(2)}_{m.group(3)}_{m.group(4) or '000000'}",
    }


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {col: METRIC_ALIASES[col] for col in df.columns if col in METRIC_ALIASES}
    return df.rename(columns=rename)


def get_evolution_info(strategy_full: str) -> dict:
    """
    根据完整策略名（含 tag）找到它在演进流程里的位置。
    优先精确匹配，找不到时按前缀匹配。
    """
    if strategy_full in EVOLUTION_ORDER:
        return EVOLUTION_ORDER[strategy_full]

    # 前缀匹配：hybrid_rerank_v2_m3 → 先匹配 hybrid_rerank_v2_m3，再 hybrid_rerank_v2，再 hybrid_rerank
    candidates = sorted(
        [k for k in EVOLUTION_ORDER if strategy_full.startswith(k)],
        key=len, reverse=True,
    )
    if candidates:
        return EVOLUTION_ORDER[candidates[0]]

    return {
        "order": 99,
        "label": strategy_full,
        "color": DEFAULT_COLOR,
        "kind": "unknown",
    }


def load_all_results() -> List[dict]:
    """
    加载所有结果 CSV。同一个 strategy_full 保留时间戳最新的。
    返回按演进顺序排好的 list。
    """
    csv_files = list(config.RESULTS_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if not f.name.startswith("summary_")
                 and not f.name.startswith("comparison_")]
    if not csv_files:
        print(f"[!] 未找到结果文件: {config.RESULTS_DIR}/*.csv")
        return []

    csv_files.sort()  # 时间戳后的会覆盖前面的
    results_map = {}
    for f in csv_files:
        meta = parse_filename(f)
        strategy_full = meta["strategy_full"]
        df = normalize_metric_columns(pd.read_csv(f, encoding="utf-8-sig"))
        info = get_evolution_info(strategy_full)
        results_map[strategy_full] = {
            "strategy_full": strategy_full,
            "df": df,
            "filename": f.name,
            "timestamp": meta["timestamp"],
            **info,
        }

    # 按演进顺序排序
    results = sorted(results_map.values(), key=lambda x: (x["order"], x["timestamp"]))

    print(f">>> 加载了 {len(results)} 个版本（按演进顺序）:")
    for r in results:
        kind_mark = {"failed": "❌", "final": "⭐", "stage": "  ", "unknown": "??"}[r["kind"]]
        print(f"    {kind_mark} [{r['label'].replace(chr(10), ' ')}]  {r['filename']}")
    return results


def compute_summary(results: List[dict]) -> pd.DataFrame:
    """Keep per-metric denominators; do not average unrelated metric populations."""
    rows=[]
    for result in results:
        row={key:result[key] for key in ("strategy_full","label","color","kind","order")}
        row["total"]=len(result["df"])
        for metric,stat in summarize(result["df"].to_dict("records"),METRIC_ORDER).items():
            row[metric]=stat["mean"]
            row[metric+"_valid"]=stat["valid"]
            row[metric+"_missing_rate"]=stat["missing_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary_table(summary: pd.DataFrame):
    print("历史逐指标描述统计（样本/有效评分不同，不生成综合排名或相对提升）")
    for _,row in summary.iterrows():
        print(row["strategy_full"])
        for metric in METRIC_ORDER:
            mean=f"{row[metric]:.4f}" if pd.notna(row[metric]) else "N/A"
            print(f"  {metric}: {mean}; valid={row[metric+'_valid']}/{row['total']}; missing={row[metric+'_missing_rate']:.1%}")


def plot_comparison(summary: pd.DataFrame, save_path: Path):
    """Four independent descriptive panels; label every available denominator."""
    summary=summary.sort_values("order").reset_index(drop=True)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    labels=[name.replace("hybrid_rerank_","rerank_").replace("baseline_v2_clean","baseline_v2") for name in summary["strategy_full"]]
    for ax,metric in zip(axes.flat,METRIC_ORDER):
        values=[float(v) if pd.notna(v) else 0 for v in summary[metric]]
        ax.bar(range(len(summary)),values,color="#446b93",width=.65)
        ax.set_xticks(range(len(summary)),labels,rotation=20,ha="right",fontsize=8)
        ax.set_ylim(0,1.16);ax.set_title(metric,fontsize=11)
        ax.spines[['top','right']].set_visible(False)
        for i,row in summary.iterrows():
            label=(f"{row[metric]:.3f}" if pd.notna(row[metric]) else "N/A")+f"\n{row[metric+'_valid']}/{row['total']} valid"
            ax.text(i,values[i]+.02,label,ha="center",fontsize=8)
    fig.suptitle("Historical RAG metrics: means with valid / total counts",fontsize=14)
    fig.text(.5,.015,"Descriptive only: changing samples and missing scores prevent an overall ranking. No fresh model evaluation.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(save_path,dpi=150);plt.close(fig)
    print(f">>> comparison: {save_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    results = load_all_results()
    if not results:
        return

    summary = compute_summary(results)
    print_summary_table(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_comparison(summary, config.RESULTS_DIR / f"comparison_{timestamp}.png")

    # 汇总 CSV
    summary_path = config.RESULTS_DIR / f"summary_{timestamp}.csv"
    summary.to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    print(f">>> 汇总表: {summary_path}")


if __name__ == "__main__":
    main()