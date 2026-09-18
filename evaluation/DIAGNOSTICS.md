# 评测审计与缺失评分处理

2026-09-18：修复报告隐藏缺失分母、未保留逐项评分失败信息的问题。历史 CSV 保持不变。

## 不消耗 API 的检查

在项目根目录，使用本机已有 `gptrag` 环境：

```bash
conda run -n gptrag python -m unittest discover -s evaluation/tests -v
conda run -n gptrag python evaluation/audit_history.py --out /tmp/rag-history-audit
```

四项回归检查覆盖：原报告缺失分母、有效零分、异常重试成功、NaN、持续失败、空输入、CSV 上下文解析，以及图表不再计算分母不同的综合排名。

## 重新评分保存的回答（会调用模型）

以下命令只准备好，执行需要已有模型配置及调用预算；不代表本轮已执行：

```bash
conda run -n gptrag python evaluation/run_evaluation.py \
  --replay-csv data/eval/results/hybrid_rerank_v2_m3_20260501_222955.csv \
  --limit 5 --metric answer_relevancy --attempts 1 \
  --tag diagnostic --out-dir /tmp/rag-rescore-diagnostic
```

`--metric` 可重复使用；省略则评分四项。`--attempts` 为指标级尝试次数，不是 HTTP 请求数，一项指标可能调用模型多次。首次初始化本地 embedding 可能需要模型文件；若缓存缺失，应先确认下载条件。

输出包括输入快照、环境/模型清单、逐项尝试 JSONL、逐条 CSV、按指标汇总 JSON。每条尝试记录 sample_id、source_row、指标、尝试序号、状态、异常类型、耗时。异常原文可能包含提供方数据，因此不直接写入报告；没有异常但返回 NaN 时保留 `non_finite_score`，不虚构具体原因。运行中断前已写入的 JSONL 会保留。

缺失值不补零。均值分母为该指标有效值数；同时显示总样本数与缺失率。跨题标准差不代表多次运行稳定性。`sample_id` 由问题与参考答案生成；重复问题需连同 source_row 区分。

## 本轮诊断边界

- 初始最小复现：`print_summary` 接收 [0.8, NaN]，输出只显示 0.8，没有 1/2 与 50%。修复后该检查通过。
- 最新历史结果前 5 条输入结构检查通过，其中第 2、3、5 条历史切题度缺失。因此这些条目的“空回答/空上下文”假设不成立。
- 当前 RAGAS 0.4.3 的切题度实现，在生成的问题全部为空时可不抛异常而直接返回 NaN；已用其实际评分函数离线复现。这只是可能路径，历史未保存响应，不能据此认定历史原因。
- 本轮没有新的真实模型评分。历史具体缺失根因仍待逐条重评分验证；也没有证明当前环境等同于 2026 年 4—5 月的实验环境。
- `visualize_results.py` 现在展示逐指标有效数和均值，移除综合分排名及不同比较分母的“提升百分比”。
