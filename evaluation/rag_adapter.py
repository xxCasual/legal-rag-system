"""
evaluation/rag_adapter.py
==========================
桥梁：把 src/ 下的 RAG 系统包装成 RAGAS 评估能用的统一接口。
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple, List
from abc import ABC, abstractmethod

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ============================================================
# 接口
# ============================================================

class RAGSystem(ABC):
    name: str

    @abstractmethod
    def query(self, question: str) -> Tuple[str, List[str]]:
        ...


# ============================================================
# 策略 1: Baseline
# ============================================================

class BaselineRAG(RAGSystem):
    name = "baseline"

    def __init__(self):
        from src.main import LegalRAGPipeline
        self.pipeline = LegalRAGPipeline(
            data_dir=str(ROOT / "data"),
            persist_dir=str(ROOT / "chroma_db"),
            verbose=False,
            rebuild_index=False,
        )

    def query(self, question: str) -> Tuple[str, List[str]]:
        answer, contexts = self.pipeline.query(question)
        if not contexts:
            contexts = [""]
        return answer, contexts


# ============================================================
# 策略 2: Hybrid (BM25 + 向量 + RRF)
# ============================================================

class HybridRAG(RAGSystem):
    name = "hybrid"

    def __init__(self):
        from src.hybrid_main import HybridLegalRAGPipeline
        self.pipeline = HybridLegalRAGPipeline(
            data_dir=str(ROOT / "data"),
            persist_dir=str(ROOT / "chroma_db"),
            verbose=False,
            rebuild_index=False,
        )

    def query(self, question: str) -> Tuple[str, List[str]]:
        answer, contexts = self.pipeline.query(question)
        if not contexts:
            contexts = [""]
        return answer, contexts


# ============================================================
# 策略 3: Hybrid + Reranker
# ============================================================

class HybridRerankRAG(RAGSystem):
    """Hybrid 检索 + BGE-reranker-base 重排序。"""
    name = "hybrid_rerank"

    def __init__(self):
        from src.hybrid_rerank_main import HybridRerankLegalRAGPipeline
        self.pipeline = HybridRerankLegalRAGPipeline(
            data_dir=str(ROOT / "data"),
            persist_dir=str(ROOT / "chroma_db"),
            verbose=False,
            rebuild_index=False,
        )

    def query(self, question: str) -> Tuple[str, List[str]]:
        answer, contexts = self.pipeline.query(question)
        if not contexts:
            contexts = [""]
        return answer, contexts


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
# CLI 测试
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="baseline")
    parser.add_argument("--question", default="保险事故发生后，被保险人应如何通知保险人？")
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
