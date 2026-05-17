# Enterprise Legal RAG Agent｜企业劳动合规 RAG Agent 平台

一个面向中国劳动合规场景的本地 RAG Agent 系统。项目基于法律法规、企业制度文档和劳动合同审查规则，提供法律问答、企业制度问答、合同风险审查、人工审批和知识库管理能力。

它不是单一的“向量检索问答 Demo”，而是一个可控的企业合规助手雏形：

- 法律问题进入法律 RAG 知识库
- 企业制度问题进入独立企业文档知识库
- 劳动合同文本进入结构化风险审查流程
- 高风险审查结果进入人工审批队列
- 本地前端控制台可以快速验收全部能力

## 功能概览

| 能力 | 说明 |
|---|---|
| 法律问答 | 基于劳动法、劳动合同法、劳动争议调解仲裁法、保险法回答法律问题 |
| 企业制度问答 | 上传员工手册、考勤制度、报销制度等企业文档后进行制度问答 |
| 法律条文管理 | 上传 `.txt` 法律条文、查看法律库、手动重建法律索引 |
| 合同风险审查 | 对劳动合同中的试用期、工资、工时、社保、解除、竞业限制等条款做风险提示 |
| 人工审批 | 高风险合同审查结果进入 `pending_review`，审批通过后才返回完整结果 |
| 前端验收控制台 | 通过 `http://127.0.0.1:8000/` 测试问答、上传、索引重建、合同审查和审批 |
| 评估体系 | 保留 RAGAS 五阶段评估和 Agent Eval 工作流评估 |

## 当前知识库

默认法律知识库包含 4 部法律文本：

- 《中华人民共和国劳动法》
- 《中华人民共和国劳动合同法》
- 《中华人民共和国劳动争议调解仲裁法》
- 《中华人民共和国保险法》

法律条文与企业制度严格分库：

| 类型 | 原始文件 | 向量索引 | 用途 |
|---|---|---|---|
| 法律条文 | `data/*.txt` | `chroma_llama_law/` | 法律问答、合同审查法律依据 |
| 企业制度 | `storage/uploads/` | `chroma_llama_company/` | 企业内部制度问答 |

## 系统架构

生产主入口是 FastAPI，统一聊天接口会先进入 LangGraph Agent 做意图识别，再路由到不同工具。

```text
用户
  ↓
FastAPI /api/chat
  ↓
LangGraph Intent Router
  ↓
┌────────────────────┬──────────────────────┬────────────────────────┬────────────────────┐
│ law_qa             │ policy_qa            │ contract_review        │ refusal            │
│ 法律 RAG 问答       │ 企业制度问答           │ 劳动合同风险审查          │ 范围外拒答           │
└────────────────────┴──────────────────────┴────────────────────────┴────────────────────┘
```

法律问答链路：

```text
用户法律问题
  ↓
问题路由：法条查询 / 法律咨询 / 知识库外
  ↓
Hybrid Retrieval：BM25 + 向量检索
  ↓
RRF 融合排序
  ↓
BGE reranker 精排
  ↓
CRAG 质检：llm / reranker / off
  ↓
法律 Prompt 生成答案，要求引用法律名称和条款编号
```

合同审查链路：

```text
合同文本
  ↓
规则抽取 7 类条款
  ↓
识别 present / missing / unclear
  ↓
计算 low / medium / high 风险等级
  ↓
必要时检索法律依据
  ↓
high 风险进入人工审批队列
```

## 核心模块

| 模块 | 路径 | 职责 |
|---|---|---|
| FastAPI 入口 | `app/main.py` | API 路由、静态前端挂载 |
| Agent 编排 | `app/agent/graph.py` | 意图识别与四路由工作流 |
| 意图分类 | `app/agent/intent_classifier.py` | 本地规则 + embedding fallback |
| 法律 RAG | `app/rag/llama_index_pipeline.py` | 生产版 LlamaIndex RAG pipeline |
| RAG 服务封装 | `app/services/rag_service.py` | 懒加载法律 RAG，并支持索引重建后重置缓存 |
| 企业文档服务 | `app/services/document_service.py` | 上传、解析、切分、索引企业制度文档 |
| 法律条文服务 | `app/services/law_document_service.py` | 上传 `.txt` 法律条文、列出法律库、重建索引 |
| 合同审查服务 | `app/services/contract_review_service.py` | 规则驱动的劳动合同风险审查 |
| 人工审批服务 | `app/services/review_service.py` | 本地 JSON 审批队列 |
| 前端控制台 | `app/static/` | 本地验收 UI |

## 技术栈

| 类别 | 选型 |
|---|---|
| API | FastAPI |
| Agent 工作流 | LangGraph |
| RAG 框架 | LlamaIndex，兼容历史 LangChain 实验入口 |
| LLM | DeepSeek compatible OpenAI-like API |
| Embedding | `BAAI/bge-small-zh-v1.5` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| 向量库 | Chroma |
| 精确检索 | BM25 |
| 评估 | RAGAS，Agent Eval |
| 前端 | 原生 HTML / CSS / JavaScript，无构建链 |

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/xxCasual/legal-rag-system.git
cd legal-rag-system

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

首次运行法律 RAG 时会下载本地 embedding 和 reranker 模型，时间取决于网络环境。

### 2. 配置环境变量

在项目根目录创建 `.env`：

```bash
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
LEGAL_RAG_LLM_MODEL=deepseek-chat
LEGAL_RAG_CRAG_MODE=reranker
```

`LEGAL_RAG_CRAG_MODE` 支持：

| 值 | 说明 |
|---|---|
| `reranker` | 默认推荐，信任本地 reranker 精排结果，速度更适合本地验收 |
| `llm` | 每个 chunk 调 LLM 做 yes/no 相关性判断，更接近早期 v5 评估链路 |
| `off` | 跳过 CRAG，适合极端调试 |

### 3. 启动服务

```bash
uvicorn app.main:app --reload
```

健康检查：

```bash
curl http://127.0.0.1:8000/api/health
```

返回：

```json
{"status":"ok"}
```

## 前端验收控制台

启动服务后打开：

```text
http://127.0.0.1:8000/
```

控制台可以直接测试：

- 法律问答
- 企业制度问答
- 范围外拒答
- 企业制度上传与列表
- 法律条文上传与列表
- 法律索引重建
- 劳动合同风险审查
- 高风险审查人工审批

前端是静态文件，由 FastAPI 直接挂载，不需要 Node、Vite 或 React 构建流程。

## API 示例

### 统一聊天

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"试用期最长多久？"}'
```

返回字段包括：

- `answer`
- `citations`
- `route`
- `intent`
- `intent_source`
- `intent_confidence`
- `tools_used`
- `result_type`
- `risk_level`
- `review_status`
- `review_id`
- `latency`

### 企业制度上传

支持 `.txt`、`.md`、`.pdf`、`.docx`：

```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload \
  -F "file=@员工手册.pdf"
```

查看企业制度文档：

```bash
curl http://127.0.0.1:8000/api/documents
```

上传后可以问：

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"公司的工资发放日是什么时候？"}'
```

### 法律条文上传与索引重建

法律条文上传目前只支持 `.txt`：

```bash
curl -X POST http://127.0.0.1:8000/api/law-documents/upload \
  -F "file=@新法律条文.txt"
```

查看法律条文：

```bash
curl http://127.0.0.1:8000/api/law-documents
```

上传法律条文后，需要手动重建法律索引：

```bash
curl -X POST http://127.0.0.1:8000/api/law-documents/rebuild-index
```

重建完成后，系统会清空已加载的法律 RAG pipeline 缓存，下一次法律问答会使用新的法律知识库。

### 劳动合同风险审查

```bash
curl -X POST http://127.0.0.1:8000/api/review/contract \
  -H "Content-Type: application/json" \
  -d '{"contract_text":"合同期限为三年。试用期一年。员工自愿放弃社保。"}'
```

合同审查覆盖 7 类条款：

- 试用期
- 合同期限
- 工资
- 工时
- 社保
- 解除 / 终止
- 竞业限制

输出内容包括：

- `risk_level`: `low` / `medium` / `high`
- `findings`: 条款级风险明细
- `evidence`: 法律依据
- `suggestions`: 修改建议
- `disclaimer`: 固定为 `仅供参考，需人工复核`
- `review_status`: `not_required` / `pending_review`

### 人工审批

高风险合同审查结果不会直接完整放出，而是进入本地审批队列。

查看待审批：

```bash
curl http://127.0.0.1:8000/api/reviews/pending
```

审批通过：

```bash
curl -X POST http://127.0.0.1:8000/api/reviews/{review_id}/approve
```

审批拒绝：

```bash
curl -X POST http://127.0.0.1:8000/api/reviews/{review_id}/reject
```

## API 清单

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 前端验收控制台 |
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/chat` | 统一 Agent 问答入口 |
| `POST` | `/api/documents/upload` | 上传企业制度文档 |
| `GET` | `/api/documents` | 查看企业制度文档 |
| `GET` | `/api/law-documents` | 查看法律条文 |
| `POST` | `/api/law-documents/upload` | 上传法律条文 `.txt` |
| `POST` | `/api/law-documents/rebuild-index` | 重建法律索引 |
| `POST` | `/api/review/contract` | 劳动合同风险审查 |
| `GET` | `/api/reviews/pending` | 查看待审批记录 |
| `POST` | `/api/reviews/{review_id}/approve` | 审批通过 |
| `POST` | `/api/reviews/{review_id}/reject` | 审批拒绝 |

## 项目结构

```text
legal-rag-system/
├── app/
│   ├── main.py                         # FastAPI 入口，API 与前端挂载
│   ├── agent/                          # LangGraph Agent 工作流
│   │   ├── graph.py                    # 四路由工作流
│   │   ├── intent_classifier.py        # 本地意图分类
│   │   └── tools.py                    # Agent 工具封装
│   ├── core/
│   │   └── config.py                   # 路径、模型、环境变量配置
│   ├── rag/
│   │   ├── llama_index_pipeline.py     # 当前生产法律 RAG pipeline
│   │   └── hybrid_rerank.py            # 历史兼容实现
│   ├── schemas/                        # Pydantic 请求/响应模型
│   ├── services/                       # 文档、法律条文、合同审查、审批服务
│   └── static/                         # 前端验收控制台
│
├── data/                               # 法律条文原始文本
│   └── eval/                           # RAGAS / Agent Eval 测试集与结果
├── storage/                            # 上传文件、审批队列、本地状态
├── chroma_llama_law/                   # 法律知识库向量索引
├── chroma_llama_company/               # 企业制度知识库向量索引
├── evaluation/                         # 评估脚本
├── tests/                              # 单元测试与 API 合同测试
├── src/                                # 历史实验和兼容入口
├── experiments/                        # 早期探索实验
└── requirements.txt
```

## 生产路径与历史路径

当前平台功能优先看 `app/`：

- `app/main.py`
- `app/agent/`
- `app/rag/llama_index_pipeline.py`
- `app/services/`
- `app/static/`

`src/` 和部分 `evaluation/` 文件主要用于历史实验、RAGAS 对比和旧入口兼容，不是当前平台运行主路径。

## 评估结果

项目保留了 RAGAS 五阶段演进记录，用于说明每一步优化的实际收益。

![五阶段演进对比](data/eval/results/comparison_20260502_164743.png)

| 版本 | 主要改动 | Faithfulness | Answer Relevancy | Context Precision | Context Recall | 平均 |
|---|---|---:|---:|---:|---:|---:|
| v1 baseline | 路由 + 向量检索 + RAG-Fusion + CRAG | 0.754 | 0.689 | 0.613 | 0.684 | 0.685 |
| v2 修路由+清洗 | 修复路由 prompt + 清洗测试集 | 0.840 | 0.799 | 0.658 | 0.772 | 0.767 |
| v3 +hybrid | BM25 + 向量 + RRF 融合检索 | 0.829 | 0.768 | 0.797 | 0.894 | 0.822 |
| v4 +rerank-base | 加入 BGE reranker-base，结果回退 | 0.804 | 0.566 | 0.795 | 0.828 | 0.748 |
| v5 +rerank-m3 | 升级到 BGE reranker-v2-m3 | 0.851 | 0.931 | 0.777 | 0.889 | 0.862 |

v5 相对 baseline 的平均分从 `0.685` 提升到 `0.862`，提升约 `25.8%`。

## 运行评估

RAGAS 评估：

```bash
python evaluation/run_evaluation.py --strategy hybrid_rerank --tag v5
python evaluation/analyze_results.py
python evaluation/visualize_results.py
```

Agent 工作流评估：

```bash
python evaluation/agent_eval.py --limit 5
```

合同审查默认可只评估规则提取和风险分级；需要完整证据链时再开启法律依据检索：

```bash
python evaluation/agent_eval.py --include-contract-evidence
```

## 测试

运行全部测试：

```bash
pytest -q
```

重点测试覆盖：

- API contract
- Agent route selection
- 企业文档上传与解析
- 法律条文上传与索引重建
- 合同审查规则
- 人工审批状态流转
- RAG pipeline 基础接口

## 设计取舍

- 当前是本地原型和验收平台，没有鉴权、租户隔离、数据库和正式权限系统。
- 人工审批队列使用本地 JSON 文件，便于演示和测试，不适合直接作为生产存储。
- 法律条文上传 v1 只支持 `.txt`，避免 PDF/DOCX 解析差异污染法律主库。
- 合同审查是规则驱动风险提示，不输出确定性法律结论。
- 当前问答是单轮为主，多轮上下文和会话记忆仍是后续方向。

## 后续方向

- 扩展 RAGAS 测试集到 200+，提升统计稳定性
- 改进跨法律对比问题的 multi-query 覆盖策略
- 增加企业后台权限、审计日志和审批人身份
- 将本地 JSON 队列迁移到数据库
- 支持多轮对话和 query rewriting
- 为法律条文管理增加版本记录和回滚能力
