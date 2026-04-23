# 基于RAG的中国劳动法智能问答系统

基于 LangChain + DeepSeek 构建的法律检索增强生成（RAG）系统。支持 Query 路由、多策略检索优化、检索结果自动质检，实现对中国劳动法领域的智能问答。

## 系统架构

```
用户提问
   │
   ▼
┌──────────────┐
│  Query路由器   │  LLM判断问题类型
└──────┬───────┘
       │
  ┌────┼────────────┐
  ▼    ▼            ▼
法条查询  法律咨询    知识库外
(直接检索) (RAG-Fusion) (礼貌拒答)
  │       │
  ▼       ▼
┌──────────────┐
│  CRAG质检     │  逐chunk判断相关性，过滤噪声
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 法律专用Prompt │  引用法条编号 + 通俗解释
│  + LLM生成    │
└──────────────┘
```

## 技术栈

- **LLM**: DeepSeek Chat（通过 OpenAI 兼容 API 调用）
- **Embedding**: BAAI/bge-small-zh-v1.5（支持中英文的轻量级模型）
- **向量数据库**: ChromaDB
- **框架**: LangChain（LCEL 管道语法）
- **知识库**: 4部中国劳动法律全文（约5.2万字，61个chunk）

## 核心功能

**1. 智能路由**：自动识别问题类型（法条查询 / 法律咨询 / 非法律问题），选择最优检索策略

**2. 多策略检索优化**：实现并对比了4种检索策略——

| 策略 | 核心思路 | 法律场景效果 |
|------|---------|------------|
| Naive RAG | 单一query直接检索 | 基线，50%噪声率 |
| RAG-Fusion | 多角度改写 + RRF排序 | 覆盖面提升150%，改写质量高 |
| HyDE | 假回答检索 | 口语化问题有优势，精确问题引入噪声 |
| CRAG | 检索后LLM质检 | 法律场景价值最高，平均过滤50%噪声 |

**3. 检索结果自动质检（CRAG）**：LLM逐chunk判断相关性，不相关的直接过滤，避免噪声法条误导回答

**4. 法律专用Prompt**：约束LLM引用具体法条编号，支持跨法规交叉引用，通俗解释法律条文

## 实验数据

### 检索质量对比（以"试用期最长多久？"为例）

| 策略 | 检索chunk数 | 包含相关法条 | 噪声过滤 |
|------|-----------|------------|---------|
| Naive RAG | 4 | 未直接命中第19条 | 无 |
| RAG-Fusion | 8 | 覆盖更广 | 无 |
| HyDE | 4 | 与Naive类似 | 无 |
| CRAG | 1 | 保留最相关chunk | 过滤3个噪声chunk |

### 关键发现

- **CRAG在法律场景下价值最高**：每道题都精准过滤噪声chunk。法律场景下一个错误的法条比没有法条更危险
- **RAG-Fusion改写质量优秀**：生成了"劳动者追索加班费的法律途径与程序"等专业化query
- **HyDE在口语化问题上有独特优势**："公司不给加班费怎么办"场景下命中了罚则第85条
- **最佳组合是RAG-Fusion + CRAG**：前者扩大覆盖面，后者过滤噪声
- **知识库外问题100%正确拒答**

## 快速开始

### 环境准备

```bash
git clone https://github.com/你的用户名/legal-rag-system.git
cd legal-rag-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Mac/Linux

# 安装依赖
pip install -r requirements.txt
```

### 配置API密钥

```bash
export DEEPSEEK_API_KEY="你的DeepSeek API Key"
```

### 运行系统

```bash
python src/main.py
```

系统会自动加载4部法律文本，构建向量索引，然后对6个测试问题进行问答，输出完整的路由判断、检索过程和最终回答。

## 项目结构

```
legal-rag-system/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖列表
├── .gitignore               # Git忽略规则
├── data/                    # 法律文本知识库
│   ├── 中华人民共和国劳动法.txt
│   ├── 中华人民共和国劳动合同法.txt
│   ├── 中华人民共和国劳动争议调解仲裁法.txt
│   └── 中华人民共和国保险法.txt
├── src/                     # 核心系统代码
│   └── main.py              # 完整RAG系统（路由+融合检索+质检+生成）
└── experiments/             # 实验过程记录
    ├── 01_naive_rag.py      # Phase 1: Naive RAG基线实验
    └── 02_strategy_comparison.py  # Phase 2: 四种策略对比实验
```

## 技术选型说明

| 决策 | 选择 | 原因 |
|------|------|------|
| Embedding模型 | bge-small-zh-v1.5 | 支持中文、轻量（90MB）、本地运行无需API |
| chunk_size | 1500 | 法条通常较长，需要比通用场景更大的chunk |
| chunk_overlap | 300 | 防止法条在chunk边界被截断 |
| 分隔符 | \\n\\n → \\n → 。 | 优先在段落和句号处切分，保持法条完整性 |
| 向量库 | ChromaDB | 轻量、适合原型验证，生产环境可换Milvus |
| 检索策略 | RAG-Fusion + CRAG | 实验验证的最优组合：覆盖面+质量控制 |

## 后续优化方向

- [ ] 接入Web搜索API，CRAG判定INCORRECT时自动回退到互联网检索
- [ ] 加入Self-RAG的忠实性检查（回答是否基于检索内容）
- [ ] 引入RAGAS框架做量化评估（Faithfulness、Answer Relevancy、Context Precision）
- [ ] 扩展知识库至更多法律领域（民法典、公司法等）
