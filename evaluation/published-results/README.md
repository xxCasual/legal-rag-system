# 历史 RAGAS 评测：公开证据

本目录是本仓库（LangChain 原型）2026-04-30 至 05-01 五个检索链路版本评测的汇总与复算结果。原始逐条结果见 [data/eval/results/](../../data/eval/results/)，均为 RAGAS 输出的 CSV，未做修改。

这些数字只属于本原型。重构版 [Enterprise_Legal_RAG_Agent](https://github.com/xxCasual/Enterprise_Legal_RAG_Agent) 的检索实现与运行环境不同，其中自带的 `_context_recall` 是自定义文本匹配指标，不是 RAGAS 的 `LLMContextRecall`，两者不能混用，历史结果也不能证明重构版效果。

## 文件

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 每个版本 × 四项指标的总数、有效数、缺失数与均值（只对有效值求均值，缺失不补零） |
| `paired-results.csv` | 两个重排版本按“问题 + 参考答案”配对的逐题 `context_recall` |
| `run-metadata.json` | 已知模型、日期与配置，以及未记录/未冻结的项目 |
| `rag-history.md` / `rag-history.json` | 复算脚本输出，含每个 CSV 的 SHA-256 与每个缺失单元格 |

## 五个版本的样本与有效分母

| 版本 | 记录数 | context_recall 有效 | 均值 | answer_relevancy 有效 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 54 | 54 | 0.684 | 4 |
| baseline_v2_clean（修路由 + 清洗测试集） | 50 | 50 | 0.772 | 2 |
| hybrid_v1（BM25 + 向量 + RRF） | 48 | 45 | 0.894 | 6 |
| hybrid_rerank_v1（bge-reranker-base） | 48 | 48 | 0.828 | 6 |
| hybrid_rerank_v2_m3（bge-reranker-v2-m3） | 48 | 48 | 0.889 | 3 |

五个版本的样本集不同：测试集在迭代中清洗过（54 → 50 → 48），baseline 的 54 条只对应 53 个不同问题。因此不同版本之间不能直接比较均值，也不计算四项指标的平均分。`answer_relevancy` 每组只有 2–6 个有效值，不据此下结论。

## 重排配置对比（0.828 → 0.889）

两个重排版本的 48 条问题与参考答案逐条一致（复算脚本断言 ID 集合相同、无重复），`context_recall` 均为 48/48 有效：

- 均值 0.828125 → 0.888889，差 0.060764
- 逐题：提高 6、不变 41、降低 1

这是两种重排配置的单轮观察对比。两版除重排模型外，候选池大小也不同，且当时未冻结全部环境与依赖，所以不能单独归因于重排模型，也没有显著性检验。这 48 题在迭代中被反复使用，属于回归集而非留出集。

## 复算

在仓库根目录运行，只读取保存的 CSV，不调用模型：

```bash
python3 evaluation/audit_history.py --out evaluation/published-results
```

2026-10-06 的输出与本目录文件一致，最后打印：

```text
{"left": "hybrid_rerank_v1_20260501_181710.csv", "right": "hybrid_rerank_v2_m3_20260501_222955.csv", "n": 48, "mean_left": 0.828125, "mean_right": 0.8888888888888888, "mean_difference": 0.06076388888888889, "improved": 6, "unchanged": 41, "worsened": 1}
```

这是“对保存结果复算”。“重新调用模型复现”需要重新检索、生成并评分，历史依赖与模型别名均未冻结，结果可能不同；本目录没有做这一步。缺失评分的处理与重评分入口见 [evaluation/DIAGNOSTICS.md](../DIAGNOSTICS.md)。
