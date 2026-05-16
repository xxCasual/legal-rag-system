# AGENTS.md

## 1. 文档目的

这是一份面向后续维护者、协作开发者和自动化 Agent 的项目接手文档。

我在通读这个仓库后，认为它不是一个单纯的“法律 RAG Demo”，而是一个已经逐步演化成企业劳动合规助手雏形的项目，包含四条清晰能力线：

1. 法律问答：基于 4 部中国劳动相关法律的 v5 RAG 主链路。
2. 企业制度问答：支持上传企业内部制度文档，单独建库并问答。
3. 劳动合同风险审查：规则驱动，必要时补法律依据。
4. 人工审批：高风险合同审查结果不直接放出，进入本地审批队列。

这个仓库已经明显区分了“生产主路径”和“历史评估兼容路径”，维护时要先认清这一点。

## 2. 一句话看项目

当前生产主路径是：

- FastAPI 入口在 `app/main.py`
- Agent 编排在 `app/agent/graph.py`
- 法律 RAG 主实现是 `app/rag/llama_index_pipeline.py`
- 合同审查、文档上传、人工审批都在 `app/services/`

历史实验和 RAGAS 兼容路径还保留在 `src/`，但它们不是平台运行时主入口。

## 3. 当前主架构

### 3.1 `/api/chat` 的真实流转

`/api/chat` 不直接调用 RAG，而是先走 LangGraph Agent：

1. `app/main.py` 的 `POST /api/chat`
2. 调用 `app.agent.run_agent_chat`
3. `app/agent/graph.py` 先做意图判断：
   - 命中企业制度关键词时，直接走 `policy_qa`
   - 命中合同审查关键词时，直接走 `contract_review`
   - 否则走 LLM 意图分类，分类结果为 `law_qa`、`policy_qa`、`contract_review` 或 `refusal`
4. 条件边进入对应节点并调用工具：
   - `search_law_articles`
   - `search_company_policy`
   - `review_labor_contract`
   - `refuse_out_of_scope`
5. 节点直接返回工具结果里的 `answer`，合同审查会额外返回结构化 `contract_review`

结论：当前 Agent 不是通用 ReAct Agent，而是一个可控的四路由 LangGraph 工作流。

### 3.2 法律问答链路

法律问答最终会走到 `app/services/rag_service.py`：

1. `RAGService` 懒加载 `LlamaIndexLegalRAGPipeline`
2. 第一次真实调用时才加载 embedding / reranker / vectorstore
3. 这样健康检查和 FastAPI 启动不会卡在大模型依赖上

`app/rag/llama_index_pipeline.py` 是生产版 LlamaIndex RAG 基座，核心步骤是：

1. 路由：法条查询 / 法律咨询 / 知识库外
2. 检索：
   - 法条查询走 Hybrid + Rerank
   - 法律咨询走 Hybrid + RAG-Fusion + Rerank
3. CRAG：
   - `llm`：逐 chunk 用 LLM 做 yes/no 相关性判断
   - `reranker`：跳过 LLM 质检，直接信任本地 reranker 排序
   - `off`：完全跳过 CRAG
4. 生成：基于过滤后的 context 输出最终答案

### 3.3 企业制度问答链路

企业制度问答与法律 RAG 完全分库：

1. 文档上传走 `DocumentService.ingest_upload`
2. 原文件保存在 `storage/uploads/`
3. 文档 registry 保存在 `storage/documents.json`
4. LlamaIndex 向量库保存在 `chroma_llama_company/`
5. collection 名是 `company_policy_docs`

问答时由 `search_company_policy()`：

1. 先 `document_service.search(query, k=4)`
2. 再用一个专门的制度问答 Prompt 生成回答

结论：企业制度能力不是走法律 RAG 的同一套知识库，它是独立索引、独立提示词、独立检索入口。

### 3.4 劳动合同风险审查链路

`app/services/contract_review_service.py` 是一个规则驱动服务，也可以通过 LangGraph 的 `contract_review` 节点作为工具调用。

它做的事情：

1. 按条款类型抽取相关段落
2. 识别条款状态：
   - `present`
   - `missing`
   - `unclear`
3. 计算风险等级：
   - `low`
   - `medium`
   - `high`
4. 输出结构化 findings、evidence、suggestions
5. 固定带 disclaimer：`仅供参考，需人工复核`

当前覆盖 7 类条款：

- 试用期
- 合同期限
- 工资
- 工时
- 社保
- 解除
- 竞业限制

这里的一个关键工程点是：

- `include_evidence=True` 时，每个条款都会通过注入的法律检索函数拉取法律依据
- `include_evidence=False` 时，完全走本地规则，不拉法律依据

这正是 Agent Eval 合同审查子集可以做到几乎瞬时的原因。

### 3.5 人工审批链路

高风险合同审查结果不会直接返回完整内容。

流程是：

1. `/api/chat` 的 `contract_review` 节点或兼容接口 `/api/review/contract` 调用 `review_labor_contract`
2. 如果 `risk_level == "high"`：
   - 调用 `review_service.create_pending_review`
   - 把完整结果写入 `storage/pending_reviews.json`
   - API 只返回 `pending_review` 状态和 `review_id`
3. 后续通过：
   - `GET /api/reviews/pending`
   - `POST /api/reviews/{review_id}/approve`
   - `POST /api/reviews/{review_id}/reject`
   完成审批

这是一个本地 JSON 持久化的人审队列，没有数据库，没有鉴权，也没有并发审批设计。

## 4. 生产路径 vs 历史兼容路径

这是这个仓库最容易看错的地方。

### 4.1 生产主路径

优先看这些文件：

- `app/main.py`
- `app/core/config.py`
- `app/agent/graph.py`
- `app/agent/tools.py`
- `app/rag/llama_index_pipeline.py`
- `app/services/*.py`

### 4.2 兼容 / 评估 / 历史实验路径

这些文件主要服务于历史版本复现、RAGAS 对比和旧入口兼容：

- `src/main.py`
- `src/hybrid_main.py`
- `src/hybrid_rerank_main.py`
- `evaluation/rag_adapter.py`
- `evaluation/run_evaluation.py`

其中 `src/hybrid_rerank_main.py` 只是历史兼容入口；当前生产实现已经迁到 `app/rag/llama_index_pipeline.py`，旧 `app/rag/hybrid_rerank.py` 只用于历史评估兼容。

维护建议：

- 做平台功能时，优先改 `app/`
- 做 RAGAS 历史对比时，再关注 `src/` 和 `evaluation/rag_adapter.py`

## 5. 配置与运行环境

### 5.1 虚拟环境

项目默认虚拟环境是项目内：

- `venv/`

运行命令时通常直接用：

```bash
venv/bin/python ...
```

### 5.2 关键环境变量

来自 `app/core/config.py` 的高频配置：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `LEGAL_RAG_LLM_MODEL`
- `LEGAL_RAG_CRAG_MODE`
- `LEGAL_RAG_DATA_DIR`
- `LEGAL_RAG_CHROMA_DIR`
- `LEGAL_RAG_COMPANY_CHROMA_DIR`
- `LEGAL_RAG_STORAGE_DIR`

### 5.3 当前 CRAG 模式设计

`LEGAL_RAG_CRAG_MODE` 支持：

- `llm`
- `reranker`
- `off`

当前默认值是 `reranker`。

这不是小细节，而是现在 law_qa 性能优化的关键开关：

- `llm` 最接近历史 v5 评估行为
- `reranker` 是当前更适合 Agent Eval 和线上问答的模式
- `off` 适合极端调试，但不建议默认使用

## 6. 数据与状态目录

这个项目不是纯只读仓库，有多处本地状态会落盘。

### 6.1 法律知识库相关

- `data/`：法律原始文本
- `chroma_llama_law/`：当前生产法律知识库向量索引
- `chroma_db/`：历史 LangChain 法律知识库向量索引

### 6.2 企业制度相关

- `storage/uploads/`：上传原文件
- `storage/documents.json`：企业制度文档 registry
- `chroma_llama_company/`：当前生产企业制度文档向量索引
- `chroma_company_docs/`：历史 LangChain 企业制度文档向量索引

### 6.3 审批相关

- `storage/pending_reviews.json`：待审批队列

### 6.4 评估相关

- `data/eval/results/`：RAGAS 输出
- `data/eval/agent_results/`：Agent Eval 输出
- `data/eval/test.json`：扩展版 Agent Eval 测试集
- `data/eval/agent_testset.json`：默认 / 较小测试集

这些目录大多已在 `.gitignore` 中忽略。

## 7. 评估体系怎么理解

这个仓库实际上有两套评估：

### 7.1 RAGAS

用途：评估法律 RAG 主链路质量。

入口：

- `evaluation/run_evaluation.py`
- `evaluation/rag_adapter.py`

它评估的是：

- faithfulness
- answer relevancy
- context precision
- context recall

适合回答的问题是：

- RAG 检索和生成质量有没有退化？
- baseline / hybrid / hybrid_rerank 哪个更好？

### 7.2 Agent Eval

用途：评估平台层 / Agent 层行为，而不是单纯法律 RAG 检索质量。

当前正式入口：

- `evaluation/agent_eval.py`

批跑入口：

- `evaluation/run_agent_eval_suite.py`

当前测试关注：

- intent_accuracy
- tool_call_accuracy
- refusal_accuracy
- risk_accuracy

适合回答的问题是：

- Agent 有没有选对工具？
- 合同审查规则命中是否稳定？
- 知识库外问题有没有正确拒答？
- 改完测试集后，分层结果怎么回归？

### 7.3 `agent_eval1.py`

仓库里还有一个 `evaluation/agent_eval1.py`。

它看起来是更激进的实验版，额外支持：

- suite filter
- warmup
- 并发执行
- cache
- raw_result
- latency summary
- `component / e2e` 两种模式

但从仓库当前整体结构看，正式对外使用的仍是 `agent_eval.py` + `run_agent_eval_suite.py` 这一组。

## 8. 我在这个项目里学到的关键事实

### 8.1 这是“平台项目”，不是只有一个 RAG 脚本

如果只盯着 `src/`，会误以为这是一个 RAG 实验仓库。
如果只盯着 `app/`，又可能忽略它保留了大量历史评估能力。

更准确的理解是：

- `app/` 是现在的平台主线
- `src/ + evaluation/rag_adapter.py` 是历史版本和 RAGAS 复现外壳

### 8.2 law_qa 慢的主要原因不是向量库，而是远程 LLM 调用链

这里最大的性能瓶颈不在 Chroma，也不在 chunk 切分，而在：

- Agent 意图分类
- 法律 RAG 内部路由
- 法律咨询场景下的 query rewrite
- 最终答案生成
- 旧 CRAG 模式下的逐 chunk LLM yes/no

所以：

- 合同审查可以通过 `include_evidence=False` 快速跑
- law_qa 即使切到 `reranker` CRAG，仍然可能偏慢

### 8.3 企业制度问答与法律问答的工程边界是清晰的

这点设计得不错：

- 法律知识库不被企业制度文档污染
- 企业制度是单独 collection
- Agent 只做路由，不强行把两者揉成一个检索器

这让后续继续扩展制度文档能力时，风险更可控。

### 8.4 合同审查并不是“LLM 审合同”，而是“规则引擎 + 可选法律依据”

这意味着：

- 它快
- 可控
- 可测试
- 适合做 Agent Eval

但也意味着：

- 对长句、口语化、变体表达的召回能力依赖规则是否覆盖到位
- 高风险规则需要持续补样本、补模式

### 8.5 当前 Agent 设计很轻

当前 Agent 没有复杂工具规划和多步推理，主要是：

- 规则优先识别企业制度问题
- 否则用一个很轻的 LLM 分类器做二选一
- 选完后就直达对应工具

这让系统简单、可控，但也意味着“知识库外拒答”仍然主要依赖法律 RAG 内部去识别，而不是在 Agent 层就明确建模一个 `refusal` 意图。

## 9. 维护时最值得注意的坑

### 9.1 不要把 `src/` 当作线上主入口

做平台功能、API、Agent、企业制度、合同审查时，优先看 `app/`。

### 9.2 改 law_qa 性能时，要分清“评估兼容”和“线上实用”

为了保持历史对比可信，有些行为需要能切回旧模式。

这也是 `LEGAL_RAG_CRAG_MODE` 存在的意义：

- `llm` 用于靠近历史 v5 行为
- `reranker` 用于当前更实用的运行模式

### 9.3 高风险合同结果不会直接返回完整答案

如果只测 `review_contract_service.review_contract()`，你看到的是完整结果。
但 API 层在高风险时会拦截为 `pending_review`。

调接口时不要把这两层混在一起理解。

### 9.4 本地状态目录很多，调试时容易受历史数据影响

典型包括：

- `chroma_db/`
- `chroma_company_docs/`
- `chroma_llama_law/`
- `chroma_llama_company/`
- `storage/documents.json`
- `storage/pending_reviews.json`
- `data/eval/agent_results/`

如果出现“结果怎么和预期不一样”，先看是不是历史索引、历史上传文档或历史审批数据在影响当前行为。

## 10. 常用命令

### 10.1 启动 API

```bash
venv/bin/python -m uvicorn app.main:app --reload
```

### 10.2 跑生产 RAG demo

```bash
venv/bin/python src/hybrid_rerank_main.py
```

### 10.3 跑 Agent Eval 单入口

```bash
venv/bin/python evaluation/agent_eval.py --testset data/eval/test.json --tag debug
```

### 10.4 跑 Agent Eval 批跑脚本

```bash
venv/bin/python evaluation/run_agent_eval_suite.py --testset data/eval/test.json --suite split --tag debug
```

```bash
venv/bin/python evaluation/run_agent_eval_suite.py --testset data/eval/test.json --suite full --tag debug
```

### 10.5 跑 RAGAS

```bash
venv/bin/python evaluation/run_evaluation.py --strategy hybrid_rerank --limit 5 --tag smoke
```

### 10.6 跑现有轻量测试

```bash
venv/bin/python tests/test_agent_eval.py
venv/bin/python tests/test_agent_eval_suite.py
venv/bin/python tests/test_agent_graph.py
venv/bin/python tests/test_api_contract.py
venv/bin/python tests/test_contract_review_service.py
venv/bin/python tests/test_document_service.py
venv/bin/python tests/test_review_service.py
venv/bin/python tests/test_config.py
venv/bin/python tests/test_llama_index_pipeline.py
venv/bin/python tests/test_crag_modes.py
venv/bin/python tests/test_v5_imports.py
```

### 10.7 静态编译检查

```bash
venv/bin/python -m compileall app src evaluation tests
```

## 11. 如果我是下一位维护者，我会先这样接手

1. 先确认 `.env`、`venv/`、`DEEPSEEK_API_KEY` 和当前 `LEGAL_RAG_LLM_MODEL`
2. 看 `app/main.py`，确认 API 面
3. 看 `app/agent/graph.py` 和 `app/agent/tools.py`，理解问答分流
4. 看 `app/rag/llama_index_pipeline.py`，理解法律 RAG 主链路
5. 看 `app/services/contract_review_service.py`，理解规则引擎
6. 跑 `tests/` 里的轻量脚本，确认环境没坏
7. 再决定是优化 law_qa、扩合同规则，还是继续扩 Agent Eval

## 12. 最后一句实话

这个项目最成熟的部分不是“API 壳子”，而是它已经有一套比较清楚的工程边界：

- 法律 RAG 和企业制度分开
- 合同审查和自由问答分开
- 生产路径和历史评估路径分开
- RAGAS 和 Agent Eval 分开

这让它很适合继续往“企业劳动合规助手”方向迭代。

真正需要持续投入的地方，主要是两类：

1. law_qa 的时延与调用链优化
2. 合同高风险规则对自然语言变体的覆盖率

如果后续继续演进，我会把这两件事视作最值得投入的主线。

## 13. LlamaIndex + LangGraph 重构计划

本轮重构目标是让生产检索层由 LlamaIndex 接管，并让 LangGraph 统一编排平台侧流程。

### 13.1 目标状态

生产主路径调整为：

1. `/api/chat` 作为统一聊天入口。
2. `app/agent/graph.py` 用 LangGraph 统一编排四类意图：
   - `law_qa`
   - `policy_qa`
   - `contract_review`
   - `refusal`
3. 法律知识库检索由 LlamaIndex 实现，保留 Hybrid、RRF、reranker、CRAG 和法律 Prompt 语义。
4. 企业制度知识库也由 LlamaIndex + 独立 Chroma collection 实现，不污染法律知识库。
5. 合同审查仍保留规则引擎和高风险人工审批，但可以作为 LangGraph 工具流的一部分被 `/api/chat` 调用。

### 13.2 索引迁移策略

新 LlamaIndex 索引使用独立目录：

- 法律知识库：`chroma_llama_law/`
- 企业制度知识库：`chroma_llama_company/`

旧目录暂不删除：

- `chroma_db/`
- `chroma_company_docs/`

这样做是为了避免 LangChain Chroma 元数据与 LlamaIndex Chroma 元数据互相污染，也方便回看历史评估结果。

### 13.3 清理策略

清理分两阶段执行：

1. 第一阶段：生产路径不再依赖旧 `app/rag/hybrid_rerank.py`，但保留历史入口和评估兼容文件。
2. 第二阶段：基于新的 Agent Eval / RAGAS 回归结果，再决定是否删除或归档：
   - `src/main.py`
   - `src/hybrid_main.py`
   - `src/hybrid_rerank_main.py`
   - `experiments/phase2_optimized.py`
   - 重复或过时的评估脚本

第一阶段不删除这些历史资产，避免破坏五阶段 RAGAS 对比链路。
