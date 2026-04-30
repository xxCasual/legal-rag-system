"""
evaluation/rag_adapter.py
==========================
桥梁：把 src/main.py 的 RAG 系统包装成 RAGAS 评估能用的统一接口。
"""

from __future__ import annotations
import os

# HuggingFace 国内镜像（必须在 import transformers/sentence_transformers 之前）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
from pathlib import Path
from typing import Tuple, List
from abc import ABC, abstractmethod

# 把项目根目录加进 sys.path，使能 from src.main import ...
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 加载 .env（项目根目录或 evaluation 目录都行）
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / "evaluation" / ".env")
except ImportError:
    pass


# ============================================================
# 接口定义
# ============================================================

class RAGSystem(ABC):
    name: str

    @abstractmethod
    def query(self, question: str) -> Tuple[str, List[str]]:
        """Returns: (answer, contexts)"""
        ...


# ============================================================
# 策略 1: Baseline - Phase 3 当前系统
# ============================================================

class BaselineRAG(RAGSystem):
    """路由 + RAG-Fusion + CRAG + 法律 Prompt（Phase 3 完整版）"""
    name = "baseline"

    def __init__(self):
        from src.main import LegalRAGPipeline

        self.pipeline = LegalRAGPipeline(
            data_dir=str(ROOT / "data"),
            persist_dir=str(ROOT / "chroma_db"),
            verbose=False,         # 评估时关闭，避免刷屏
            rebuild_index=False,   # 复用已构建的索引
        )

    def query(self, question: str) -> Tuple[str, List[str]]:
        answer, contexts = self.pipeline.query(question)
        # 路由到"知识库外"时 contexts 为空，给占位避免 RAGAS 评估崩溃
        if not contexts:
            contexts = [""]
        return answer, contexts


# ============================================================
# 策略 2: Hybrid - Baseline + BM25/向量混合检索
# ============================================================

class HybridRAG(RAGSystem):
    """等你实施 hybrid search 后再填"""
    name = "hybrid"

    def __init__(self):
        raise NotImplementedError(
            "等你实施 hybrid search 后再填这里。\n"
            "建议: from src.hybrid_main import HybridLegalRAGPipeline"
        )

    def query(self, question: str) -> Tuple[str, List[str]]:
        raise NotImplementedError


# ============================================================
# 策略 3: Hybrid + Reranker
# ============================================================

class HybridRerankRAG(RAGSystem):
    """等你实施 reranker 后再填"""
    name = "hybrid_rerank"

    def __init__(self):
        raise NotImplementedError("等实施 reranker 后填充")

    def query(self, question: str) -> Tuple[str, List[str]]:
        raise NotImplementedError


# ============================================================
# 工厂
# ============================================================

STRATEGIES = {
    "baseline": BaselineRAG,
    "hybrid": HybridRAG,
    "hybrid_rerank": HybridRerankRAG,
}


def get_rag_system(strategy_name: str) -> RAGSystem:
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"未知策略: {strategy_name}. 可选: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[strategy_name]()


# ============================================================
# 命令行：本地快速测试
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="baseline")
    parser.add_argument("--question", default="未签书面劳动合同有什么后果？")
    args = parser.parse_args()

    rag = get_rag_system(args.strategy)
    answer, contexts = rag.query(args.question)

    print(f"\n[策略] {args.strategy}")
    print(f"[问题] {args.question}")
    print(f"\n[答案]\n{answer}")
    print(f"\n[检索到 {len(contexts)} 个 chunk]")
    for i, c in enumerate(contexts, 1):
        print(f"\n--- Chunk {i} (长度 {len(c)}) ---")
        print(c[:300] + ("..." if len(c) > 300 else ""))
