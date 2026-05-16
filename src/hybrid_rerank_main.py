"""
Compatibility entrypoint for the production v5 RAG pipeline.

The implementation now lives in app.rag.hybrid_rerank so the enterprise
application can build on a single production RAG base. This module keeps the
old import path and CLI demo behavior working for existing evaluation scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.hybrid_rerank import (  # noqa: E402
    HybridRerankLegalRAGPipeline,
    _run_demo,
)

__all__ = ["HybridRerankLegalRAGPipeline"]


if __name__ == "__main__":
    _run_demo()
