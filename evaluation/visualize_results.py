"""
visualize_results.py
=====================
法律 RAG 五阶段演进对比可视化。

设计：
1. 按"项目演进顺序"排版本（baseline → v2_clean → hybrid → rerank_v1 → rerank_v2_m3），
   而不是按文件名字母序——讲清楚"做了哪些事，每一步带来什么变化"
2. 每个版本独立颜色，失败的实验用红色标注
3. 4 张子图组合：
   - 左上：演进折线图（核心讲故事图）
   - 右上：分组柱状图
   - 左下：雷达图（baseline vs 最终版）
   - 右下：综合排名 + 失败实验标注
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import config

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
    "hybrid_rerank_v1":     {"order": 3, "label": "v4\n+rerank-base\n(失败)", "color": "#E74C3C", "kind": "failed"},
    "hybrid_rerank":        {"order": 3, "label": "v4\n+rerank-base\n(失败)", "color": "#E74C3C", "kind": "failed"},
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
    m = re.match(r"^(.+?)_(\d{8})_(\d{6})$", stem)
    if not m:
        return {"strategy_full": stem, "timestamp": "unknown"}
    return {
        "strategy_full": m.group(1),
        "timestamp": f"{m.group(2)}_{m.group(3)}",
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
    """每个版本的指标均值矩阵，行按演进顺序"""
    rows = []
    for r in results:
        row = {
            "strategy_full": r["strategy_full"],
            "label": r["label"],
            "color": r["color"],
            "kind": r["kind"],
            "order": r["order"],
        }
        for metric in METRIC_ORDER:
            row[metric] = (
                r["df"][metric].dropna().astype(float).mean()
                if metric in r["df"].columns else np.nan
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    metric_cols = [c for c in METRIC_ORDER if c in df.columns]
    df["overall"] = df[metric_cols].mean(axis=1)
    return df


# ============================================================
# 控制台输出
# ============================================================

def print_summary_table(summary: pd.DataFrame):
    print("\n" + "=" * 90)
    print("演进对比表（按时间顺序）")
    print("=" * 90)

    metric_cols = [c for c in METRIC_ORDER if c in summary.columns]
    display_df = summary[["label"] + metric_cols].copy()
    display_df["label"] = display_df["label"].str.replace("\n", " ")
    display_df["平均"] = display_df[metric_cols].mean(axis=1)

    rename = {c: METRIC_DISPLAY[c].replace("\n", " ") for c in metric_cols}
    rename["label"] = "版本"
    display_df = display_df.rename(columns=rename)
    print(display_df.round(4).to_string(index=False))

    # 与 baseline 对比
    base = summary[summary["strategy_full"].isin(["baseline"])]
    if len(base) > 0 and len(summary) > 1:
        print("\n相对 v1 baseline 的提升:")
        print("-" * 90)
        base_row = base.iloc[0]
        for _, row in summary.iterrows():
            if row["strategy_full"] == "baseline":
                continue
            label = row["label"].replace("\n", " ")
            kind = row["kind"]
            mark = "❌" if kind == "failed" else ("⭐" if kind == "final" else "  ")
            print(f"  {mark} [{label}]")
            for metric in metric_cols:
                diff = row[metric] - base_row[metric]
                pct = diff / base_row[metric] * 100 if base_row[metric] != 0 else 0
                sign = "+" if diff >= 0 else ""
                print(f"      {METRIC_DISPLAY[metric].replace(chr(10), ' '):<32} "
                      f"{sign}{diff:+.4f} ({sign}{pct:+.1f}%)")
    print("=" * 90)


# ============================================================
# 可视化
# ============================================================

def plot_comparison(summary: pd.DataFrame, save_path: Path):
    """
    4 子图布局：
        左上 (gs[0,0]): 演进折线图（每个指标一条线）
        右上 (gs[0,1]): 分组柱状图（按版本横排，每组 4 个指标）
        左下 (gs[1,0]): 雷达图（baseline vs 最终版）
        右下 (gs[1,1]): 综合排名 + 演进路径
    """
    summary = summary.sort_values("order").reset_index(drop=True)
    metric_cols = [c for c in METRIC_ORDER if c in summary.columns]
    metric_labels = [METRIC_DISPLAY[m] for m in metric_cols]

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3,
                           left=0.07, right=0.97, top=0.93, bottom=0.07)

    colors = summary["color"].tolist()
    labels = summary["label"].tolist()
    kinds = summary["kind"].tolist()

    # ============================================================
    # 子图 1（左上）：演进折线图 ⭐ 核心叙事图
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # 4 个指标各一条线
    metric_line_colors = {
        "context_recall":     "#27AE60",
        "context_precision":  "#3498DB",
        "faithfulness":       "#E67E22",
        "answer_relevancy":   "#9B59B6",
    }
    metric_markers = {
        "context_recall":     "o",
        "context_precision":  "s",
        "faithfulness":       "^",
        "answer_relevancy":   "D",
    }

    x = np.arange(len(summary))
    for metric in metric_cols:
        vals = summary[metric].values
        ax1.plot(
            x, vals,
            marker=metric_markers[metric], markersize=10, linewidth=2.2,
            color=metric_line_colors[metric],
            label=METRIC_DISPLAY[metric].replace("\n", " "),
        )
        # 在每个点上标注数值
        for xi, v in zip(x, vals):
            ax1.annotate(
                f"{v:.3f}", (xi, v),
                textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=8, color=metric_line_colors[metric],
                fontweight="bold",
            )

    # 标记失败版本
    for i, kind in enumerate(kinds):
        if kind == "failed":
            ax1.axvspan(i - 0.4, i + 0.4, alpha=0.12, color="red", zorder=0)
            ax1.text(i, 0.42, "失败实验", ha="center", color="#C0392B",
                     fontsize=9, fontweight="bold")
        elif kind == "final":
            ax1.axvspan(i - 0.4, i + 0.4, alpha=0.12, color="green", zorder=0)
            ax1.text(i, 0.42, "最终版本", ha="center", color="#1E8449",
                     fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("分数", fontsize=11)
    ax1.set_ylim(0.4, 1.05)
    ax1.set_title("各指标随版本演进的变化", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # 子图 2（右上）：分组柱状图（按指标分组，每组多版本对比）
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    n_versions = len(summary)
    x_metric = np.arange(len(metric_cols))
    width = 0.8 / n_versions

    for i, (_, row) in enumerate(summary.iterrows()):
        vals = [row[m] for m in metric_cols]
        offset = (i - n_versions / 2 + 0.5) * width
        edge_color = "black" if row["kind"] in ("final", "failed") else "white"
        edge_width = 1.5 if row["kind"] in ("final", "failed") else 0.5
        bars = ax2.bar(
            x_metric + offset, vals, width,
            color=row["color"], alpha=0.92,
            edgecolor=edge_color, linewidth=edge_width,
            label=row["label"].replace("\n", " "),
        )
        for bar, v in zip(bars, vals):
            if not pd.isna(v):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.012,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                )

    ax2.set_xticks(x_metric)
    ax2.set_xticklabels(metric_labels, fontsize=9)
    ax2.set_ylabel("分数", fontsize=11)
    ax2.set_ylim(0, 1.12)
    ax2.set_title("各指标上的版本对比", fontsize=13, fontweight="bold", pad=12)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
               ncol=min(n_versions, 5), fontsize=8, frameon=False)
    ax2.grid(True, alpha=0.3, axis="y")

    # ============================================================
    # 子图 3（左下）：雷达图（baseline vs 最终版本，避免胡成一片）
    # ============================================================
    ax3 = fig.add_subplot(gs[1, 0], projection="polar")

    angles = np.linspace(0, 2 * np.pi, len(metric_cols), endpoint=False).tolist()
    angles += angles[:1]

    # 只画 baseline + 最终版（如果有失败的也画上）
    versions_to_plot = []
    for _, row in summary.iterrows():
        if row["kind"] in ("stage",) and row["order"] == 0:
            versions_to_plot.append(row)  # baseline
        elif row["kind"] == "failed":
            versions_to_plot.append(row)  # 失败版本，对比用
        elif row["kind"] == "final":
            versions_to_plot.append(row)  # 最终版本

    if not versions_to_plot:
        # 退路：画前两个
        versions_to_plot = [summary.iloc[0], summary.iloc[-1]]

    for row in versions_to_plot:
        vals = [row[m] for m in metric_cols]
        vals += vals[:1]
        lw = 2.8 if row["kind"] == "final" else 2.0
        ax3.plot(angles, vals, "o-", linewidth=lw, color=row["color"],
                 label=row["label"].replace("\n", " "))
        ax3.fill(angles, vals, alpha=0.18, color=row["color"])

    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels([m.replace("\n", " ") for m in metric_labels], fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax3.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax3.set_title("关键版本能力雷达图\n(baseline vs 失败版本 vs 最终版本)",
                  fontsize=12, fontweight="bold", pad=20)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=8)
    ax3.grid(True, alpha=0.4)

    # ============================================================
    # 子图 4（右下）：综合排名条形图
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 1])

    sorted_summary = summary.sort_values("overall", ascending=True)

    bars = ax4.barh(
        range(len(sorted_summary)), sorted_summary["overall"].values,
        color=sorted_summary["color"].tolist(), alpha=0.92,
        edgecolor=["black" if k in ("final", "failed") else "white"
                   for k in sorted_summary["kind"]],
        linewidth=[1.5 if k in ("final", "failed") else 0.5
                   for k in sorted_summary["kind"]],
    )
    ax4.set_yticks(range(len(sorted_summary)))
    ax4.set_yticklabels(
        [lbl.replace("\n", " ") for lbl in sorted_summary["label"]], fontsize=10,
    )

    # 标分数 + 排名状态
    for i, (bar, (_, row)) in enumerate(zip(bars, sorted_summary.iterrows())):
        v = row["overall"]
        text_extra = ""
        if row["kind"] == "final":
            text_extra = "  ⭐ 最终版"
        elif row["kind"] == "failed":
            text_extra = "  ❌ 失败实验"
        ax4.text(v + 0.012, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}{text_extra}", va="center", fontsize=10, fontweight="bold")

    ax4.set_xlabel("4 个指标的平均分", fontsize=11)
    ax4.set_xlim(0, 1.05)
    ax4.set_title("综合排名（4 指标平均）", fontsize=13, fontweight="bold", pad=12)
    ax4.grid(True, alpha=0.3, axis="x")

    # 总标题
    final_score = summary[summary["kind"] == "final"]["overall"]
    base_score = summary[summary["strategy_full"] == "baseline"]["overall"]
    if len(final_score) > 0 and len(base_score) > 0:
        improvement = (final_score.iloc[0] - base_score.iloc[0]) / base_score.iloc[0] * 100
        title_extra = f"  |  最终版相对 baseline 提升 {improvement:+.1f}%"
    else:
        title_extra = ""

    fig.suptitle(
        f"法律 RAG 五阶段演进评估对比{title_extra}",
        fontsize=15, fontweight="bold", y=0.985,
    )

    plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"\n>>> 对比图已保存: {save_path}")


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
    summary[["label", "strategy_full", "kind"] + METRIC_ORDER + ["overall"]].to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )
    print(f">>> 汇总表: {summary_path}")


if __name__ == "__main__":
    main()