# RAG 历史数据复算

本报告只重算保存的 CSV；未重新检索、生成或调用模型。历史错误原因未被保存，缺失统一标注 historical_missing_unknown。

| 文件 | 指标 | 有效/总数 | 缺失率 | 均值 |
| --- | --- | --- | --- | --- |
| baseline_20260430_202900.csv | faithfulness | 45/54 | 16.7% | 0.754378 |
| baseline_20260430_202900.csv | answer_relevancy | 4/54 | 92.6% | 0.689038 |
| baseline_20260430_202900.csv | llm_context_precision_with_reference | 53/54 | 1.9% | 0.613208 |
| baseline_20260430_202900.csv | context_recall | 54/54 | 0.0% | 0.683642 |
| baseline_v2_clean_20260430_205714.csv | faithfulness | 39/50 | 22.0% | 0.839839 |
| baseline_v2_clean_20260430_205714.csv | answer_relevancy | 2/50 | 96.0% | 0.799181 |
| baseline_v2_clean_20260430_205714.csv | llm_context_precision_with_reference | 46/50 | 8.0% | 0.657609 |
| baseline_v2_clean_20260430_205714.csv | context_recall | 50/50 | 0.0% | 0.771667 |
| hybrid_rerank_v1_20260501_181710.csv | faithfulness | 36/48 | 25.0% | 0.804253 |
| hybrid_rerank_v1_20260501_181710.csv | answer_relevancy | 6/48 | 87.5% | 0.565451 |
| hybrid_rerank_v1_20260501_181710.csv | llm_context_precision_with_reference | 44/48 | 8.3% | 0.795455 |
| hybrid_rerank_v1_20260501_181710.csv | context_recall | 48/48 | 0.0% | 0.828125 |
| hybrid_rerank_v2_m3_20260501_222955.csv | faithfulness | 44/48 | 8.3% | 0.851325 |
| hybrid_rerank_v2_m3_20260501_222955.csv | answer_relevancy | 3/48 | 93.8% | 0.930563 |
| hybrid_rerank_v2_m3_20260501_222955.csv | llm_context_precision_with_reference | 46/48 | 4.2% | 0.777174 |
| hybrid_rerank_v2_m3_20260501_222955.csv | context_recall | 48/48 | 0.0% | 0.888889 |
| hybrid_v1_20260501_144846.csv | faithfulness | 35/48 | 27.1% | 0.828842 |
| hybrid_v1_20260501_144846.csv | answer_relevancy | 6/48 | 87.5% | 0.767714 |
| hybrid_v1_20260501_144846.csv | llm_context_precision_with_reference | 48/48 | 0.0% | 0.796875 |
| hybrid_v1_20260501_144846.csv | context_recall | 45/48 | 6.2% | 0.894444 |

## 同题、同参考答案的召回对比

两组各 48 条且召回评分完整：0.828125 → 0.888889，差 0.060764。
逐题：提高 6、不变 41、降低 1。

仅是历史两种重排配置的观察对比；未冻结所有环境变量，不能单独归因于重排模型。48 题已用于迭代，是回归集。

不计算四指标平均的“综合分”；切题度每组仅 2–6 个有效值，不能用其标准差描述运行稳定性。
