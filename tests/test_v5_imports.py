"""
Smoke test for the v5-first RAG refactor.

Run with:
    python tests/test_v5_imports.py
"""

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
_install_module_stub("langchain_community.vectorstores", Chroma=_Dummy)
_install_module_stub("langchain_community.retrievers", BM25Retriever=_Dummy)
_install_module_stub(
    "langchain_text_splitters", RecursiveCharacterTextSplitter=_Dummy
)
_install_module_stub("langchain_huggingface", HuggingFaceEmbeddings=_Dummy)
_install_module_stub("langchain_chroma", Chroma=_Dummy)
_install_module_stub("langchain_core")
_install_module_stub("langchain_core.prompts", ChatPromptTemplate=_Dummy)
_install_module_stub("langchain_core.output_parsers", StrOutputParser=_Dummy)
_install_module_stub("langchain_core.documents", Document=_Dummy)
_install_module_stub("langchain_openai", ChatOpenAI=_Dummy)

from app.rag import HybridRerankLegalRAGPipeline as AppPipeline  # noqa: E402
from src.hybrid_rerank_main import (  # noqa: E402
    HybridRerankLegalRAGPipeline as CompatPipeline,
)


def test_v5_pipeline_import_paths_match() -> None:
    assert AppPipeline is CompatPipeline


if __name__ == "__main__":
    test_v5_pipeline_import_paths_match()
    print("v5 import compatibility ok")
