"""Production RAG implementation for the enterprise legal assistant platform."""

from app.rag.llama_index_pipeline import LlamaIndexLegalRAGPipeline

__all__ = ["HybridRerankLegalRAGPipeline", "LlamaIndexLegalRAGPipeline"]


def __getattr__(name: str):
    if name == "HybridRerankLegalRAGPipeline":
        from app.rag.hybrid_rerank import HybridRerankLegalRAGPipeline

        return HybridRerankLegalRAGPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
