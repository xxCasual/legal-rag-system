"""Unit checks for configurable CRAG behavior."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Dummy:
    pass


def _install_module_stub(name: str, **attrs: object) -> None:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules.setdefault(name, module)


_install_module_stub("langchain_community")
_install_module_stub(
    "langchain_community.document_loaders",
    TextLoader=_Dummy,
    DirectoryLoader=_Dummy,
)
_install_module_stub("langchain_community.retrievers", BM25Retriever=_Dummy)
_install_module_stub("langchain_text_splitters", RecursiveCharacterTextSplitter=_Dummy)
_install_module_stub("langchain_huggingface", HuggingFaceEmbeddings=_Dummy)
_install_module_stub("langchain_chroma", Chroma=_Dummy)
_install_module_stub("langchain_core")
_install_module_stub("langchain_core.prompts", ChatPromptTemplate=_Dummy)
_install_module_stub("langchain_core.output_parsers", StrOutputParser=_Dummy)
_install_module_stub("langchain_core.documents", Document=_Dummy)
_install_module_stub("langchain_openai", ChatOpenAI=_Dummy)

from app.rag.hybrid_rerank import (  # noqa: E402
    HybridRerankLegalRAGPipeline,
    normalize_crag_mode,
)


class _Doc:
    def __init__(self, text: str):
        self.page_content = text
        self.metadata = {}


class _FailingGradeChain:
    def invoke(self, _: object) -> str:
        raise AssertionError("grade_chain should not be called")


class _KeywordGradeChain:
    def invoke(self, payload: dict[str, str]) -> str:
        return "yes" if "相关" in payload["document"] else "no"


def _pipeline_with_mode(mode: str) -> HybridRerankLegalRAGPipeline:
    pipeline = object.__new__(HybridRerankLegalRAGPipeline)
    pipeline.crag_mode = mode
    pipeline.verbose = False
    pipeline.grade_chain = _FailingGradeChain()
    return pipeline


def test_normalize_crag_mode_accepts_supported_values() -> None:
    assert normalize_crag_mode("LLM") == "llm"
    assert normalize_crag_mode(" reranker ") == "reranker"
    assert normalize_crag_mode("off") == "off"


def test_normalize_crag_mode_rejects_invalid_values() -> None:
    try:
        normalize_crag_mode("remote")
    except ValueError as exc:
        assert "llm" in str(exc)
        assert "reranker" in str(exc)
        assert "off" in str(exc)
    else:
        raise AssertionError("invalid CRAG mode should be rejected")


def test_reranker_crag_mode_skips_llm_grader() -> None:
    docs = [_Doc("第一条"), _Doc("第二条")]
    pipeline = _pipeline_with_mode("reranker")

    assert pipeline._crag_filter("问题", docs) == docs


def test_off_crag_mode_skips_llm_grader() -> None:
    docs = [_Doc("第一条"), _Doc("第二条")]
    pipeline = _pipeline_with_mode("off")

    assert pipeline._crag_filter("问题", docs) == docs


def test_llm_crag_mode_keeps_yes_chunks() -> None:
    docs = [_Doc("相关条文"), _Doc("无关条文")]
    pipeline = _pipeline_with_mode("llm")
    pipeline.grade_chain = _KeywordGradeChain()

    assert pipeline._crag_filter("问题", docs) == docs[:1]


if __name__ == "__main__":
    test_normalize_crag_mode_accepts_supported_values()
    test_normalize_crag_mode_rejects_invalid_values()
    test_reranker_crag_mode_skips_llm_grader()
    test_off_crag_mode_skips_llm_grader()
    test_llm_crag_mode_keeps_yes_chunks()
    print("crag mode tests ok")
