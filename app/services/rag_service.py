"""Service wrapper around the production v5 RAG pipeline."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any, Dict, List


class RAGService:
    """Lazy singleton wrapper for the v5 RAG pipeline.

    The heavy LangChain, embedding, and reranker dependencies are imported only
    when the first chat request arrives. This keeps FastAPI startup and health
    checks lightweight.
    """

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._lock = Lock()

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    from app.rag import LlamaIndexLegalRAGPipeline

                    self._pipeline = LlamaIndexLegalRAGPipeline(
                        verbose=False,
                        rebuild_index=False,
                    )
        return self._pipeline

    def chat(self, query: str) -> Dict[str, Any]:
        started_at = time.perf_counter()
        result = self._get_pipeline().query_with_details(query)
        latency = round(time.perf_counter() - started_at, 3)

        contexts: List[str] = result.get("contexts") or []
        return {
            "answer": result.get("answer", ""),
            "citations": contexts,
            "route": result.get("route", ""),
            "latency": latency,
        }


rag_service = RAGService()
