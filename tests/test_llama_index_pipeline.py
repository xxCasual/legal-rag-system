"""LlamaIndex pipeline unit checks without loading real models."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag.llama_index_pipeline import (  # noqa: E402
    LlamaIndexLegalRAGPipeline,
    normalize_crag_mode,
)
from app.services.rag_service import RAGService  # noqa: E402
import app.rag as rag_module  # noqa: E402


class _Node:
    def __init__(self, text: str):
        self._text = text

    def get_content(self) -> str:
        return self._text


class _NodeWithScore:
    def __init__(self, text: str, score: float = 1.0):
        self.node = _Node(text)
        self.score = score


class _StaticLLM:
    def complete(self, prompt: str):
        if "类型：" in prompt:
            return SimpleNamespace(text="法条查询")
        return SimpleNamespace(text="基于条文的答案")


class _StaticRetriever:
    def retrieve(self, _: str):
        return [_NodeWithScore("劳动合同法 第十九条 试用期规定", 1.0)]


class _FailingLLM:
    def complete(self, prompt: str):
        raise TimeoutError("llm timed out")


class _SlowLLM:
    def complete(self, prompt: str):
        deadline = time.perf_counter() + 2
        while time.perf_counter() < deadline:
            try:
                time.sleep(0.2)
            except TimeoutError:
                continue
        return SimpleNamespace(text="迟到的答案")


def test_normalize_crag_mode_matches_supported_legacy_values() -> None:
    assert normalize_crag_mode("LLM") == "llm"
    assert normalize_crag_mode(" reranker ") == "reranker"
    assert normalize_crag_mode("off") == "off"


def test_query_with_details_keeps_service_contract() -> None:
    pipeline = object.__new__(LlamaIndexLegalRAGPipeline)
    pipeline.crag_mode = "reranker"
    pipeline.verbose = False
    pipeline.llm = _StaticLLM()
    pipeline.vector_retriever = _StaticRetriever()
    pipeline.bm25_retriever = _StaticRetriever()
    pipeline.reranker = None
    pipeline.rerank_top_n = 4
    pipeline.rrf_k_const = 60

    result = pipeline.query_with_details("试用期最长多久？")

    assert result["route"] == "法条查询"
    assert result["answer"] == "基于条文的答案"
    assert result["contexts"] == ["劳动合同法 第十九条 试用期规定"]
    assert result["crag_mode"] == "reranker"


def test_law_route_uses_heuristic_before_llm() -> None:
    pipeline = object.__new__(LlamaIndexLegalRAGPipeline)
    pipeline.llm = _FailingLLM()
    pipeline.verbose = False

    assert pipeline._route("试用期最长多久？") == "法条查询"


def test_answer_falls_back_to_context_summary_when_llm_fails() -> None:
    pipeline = object.__new__(LlamaIndexLegalRAGPipeline)
    pipeline.llm = _FailingLLM()
    pipeline.verbose = False
    contexts = ["劳动合同法 第十九条 试用期不得超过六个月。"]

    answer = pipeline._answer("试用期最长多久？", contexts)

    assert "劳动合同法 第十九条" in answer
    assert "根据检索到的法律条文" in answer


def test_answer_falls_back_when_llm_call_exceeds_timeout() -> None:
    pipeline = object.__new__(LlamaIndexLegalRAGPipeline)
    pipeline.llm = _SlowLLM()
    pipeline.verbose = False
    pipeline.llm_timeout_seconds = 0.05
    contexts = ["劳动合同法 第十九条 试用期不得超过六个月。"]

    started_at = time.perf_counter()
    answer = pipeline._answer("试用期最长多久？", contexts)

    assert time.perf_counter() - started_at < 0.5
    assert "劳动合同法 第十九条" in answer


def test_generate_queries_returns_empty_list_when_llm_fails() -> None:
    pipeline = object.__new__(LlamaIndexLegalRAGPipeline)
    pipeline.llm = _FailingLLM()
    pipeline.verbose = False

    assert pipeline._generate_queries("被拖欠工资怎么办？") == []


def test_rag_service_lazy_loads_llama_index_pipeline() -> None:
    class _FakePipeline:
        def __init__(self, verbose: bool, rebuild_index: bool):
            self.verbose = verbose
            self.rebuild_index = rebuild_index

        def query_with_details(self, query: str):
            assert query == "试用期最长多久？"
            return {
                "answer": "答案",
                "contexts": ["依据"],
                "route": "法条查询",
            }

    original = getattr(rag_module, "LlamaIndexLegalRAGPipeline", None)
    rag_module.LlamaIndexLegalRAGPipeline = _FakePipeline
    try:
        result = RAGService().chat("试用期最长多久？")
    finally:
        if original is None:
            delattr(rag_module, "LlamaIndexLegalRAGPipeline")
        else:
            rag_module.LlamaIndexLegalRAGPipeline = original

    assert result["answer"] == "答案"
    assert result["citations"] == ["依据"]
    assert result["route"] == "法条查询"


if __name__ == "__main__":
    test_normalize_crag_mode_matches_supported_legacy_values()
    test_query_with_details_keeps_service_contract()
    test_law_route_uses_heuristic_before_llm()
    test_answer_falls_back_to_context_summary_when_llm_fails()
    test_answer_falls_back_when_llm_call_exceeds_timeout()
    test_generate_queries_returns_empty_list_when_llm_fails()
    test_rag_service_lazy_loads_llama_index_pipeline()
    print("llama index pipeline ok")
