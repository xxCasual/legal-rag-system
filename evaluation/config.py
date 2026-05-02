"""
config.py
=========
集中配置。修改这里就能切换模型、改路径，不用动其他代码。

LLM: DeepSeek-V3 (deepseek-chat)
Embeddings: 本地 BGE-large-zh (中文场景质量最好，免费)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 路径
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR.parent / "data" / "eval" / "results"
TESTSET_PATH = BASE_DIR.parent / "data" / "eval" / "testset.json"
LEGAL_DOCS_DIR = BASE_DIR.parent / "data"


# 你的法律文档目录（指向 legal-rag-system 项目里的劳动法文档）
# 修改成你本地的实际路径
LEGAL_DOCS_DIR = Path(os.getenv("LEGAL_DOCS_DIR", "../legal-rag-system/data/laws"))

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# API 配置 - DeepSeek
# ============================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

if not DEEPSEEK_API_KEY:
    print("[!] 警告: 未设置 DEEPSEEK_API_KEY 环境变量")
    print("    在 .env 文件里加: DEEPSEEK_API_KEY=sk-xxxxx")


# ============================================================
# 模型配置
# ============================================================

# 测试集生成器 LLM
# - deepseek-chat (V3): 推荐，性价比高，JSON 输出稳定
# - deepseek-reasoner (R1): 不推荐，会输出 reasoning 内容影响 RAGAS 解析，且贵
GENERATOR_LLM_MODEL = "deepseek-chat"

# 评估器 LLM
EVALUATOR_LLM_MODEL = "deepseek-chat"

# Embedding 模型 - 本地 BGE，中文场景明显优于 OpenAI text-embedding-3
# 首次使用会自动下载约 1.3GB 到 ~/.cache/huggingface/
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"

# 如果你显存不够或不想本地跑，可以换成更小的：
# EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 约 100MB


# ============================================================
# 测试集生成配置
# ============================================================

# 测试集大小。建议 30-100，太少统计意义不够，太多成本高
TESTSET_SIZE = 50

# 问题类型分布
QUERY_DISTRIBUTION = {
    "single_hop": 0.5,
    "multi_hop_abstract": 0.25,
    "multi_hop_specific": 0.25,
}


# ============================================================
# 评估配置
# ============================================================

# RAGAS 评估的 LLM 并发数
MAX_CONCURRENT = 4

# LLM 温度。评估场景必须为 0，保证可复现性
LLM_TEMPERATURE = 0


# ============================================================
# 工厂函数：构造 LangChain LLM 和 Embeddings 对象
# ============================================================

def get_llm(model: str = None, json_mode: bool = True):
    """
    构造 DeepSeek LLM (走 OpenAI 兼容接口)。

    Args:
        model: 模型名
        json_mode: 是否强制 JSON 输出格式。True 时模型输出严格符合 JSON 标准，
                   不会出现 \\' 这种非法转义。RAGAS 大部分场景需要 JSON，开启更稳。
    """
    from langchain_openai import ChatOpenAI
    kwargs = dict(
        model=model or EVALUATOR_LLM_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=LLM_TEMPERATURE,
        max_retries=5,
        timeout=180,
    )
    if json_mode:
        # DeepSeek 兼容 OpenAI 的 response_format JSON 模式
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    return ChatOpenAI(**kwargs)


def get_embeddings():
    """构造本地 BGE embeddings。"""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # bge-zh 推荐这两个设置
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"},  # 有 GPU 改成 "cuda"
    )
