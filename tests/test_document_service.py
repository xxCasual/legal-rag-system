"""DocumentService checks without loading embedding or vector models."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.document_service import DocumentService  # noqa: E402
from app.core.config import settings  # noqa: E402


class _FakeVectorStore:
    def __init__(self) -> None:
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)


def test_txt_ingest_persists_metadata() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        vectorstore = _FakeVectorStore()
        service = DocumentService(
            uploads_dir=root / "uploads",
            registry_path=root / "documents.json",
            persist_dir=root / "chroma",
        )
        service._get_vectorstore = lambda: vectorstore  # type: ignore[method-assign]

        record = service.ingest_upload(
            "employee_policy.txt",
            "迟到三次记一次书面提醒。工资发放日为每月十日。".encode("utf-8"),
        )

        assert record["file_name"] == "employee_policy.txt"
        assert record["source_type"] == "txt"
        assert record["chunk_count"] >= 1
        assert len(vectorstore.documents) == record["chunk_count"]

        first = vectorstore.documents[0]
        assert first.metadata["doc_id"] == record["doc_id"]
        assert first.metadata["file_name"] == "employee_policy.txt"
        assert first.metadata["source_type"] == "txt"
        assert first.metadata["chunk_id"] == 0

        registry = json.loads((root / "documents.json").read_text(encoding="utf-8"))
        assert registry[0]["doc_id"] == record["doc_id"]


def test_md_ingest_and_list_documents() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        service = DocumentService(
            uploads_dir=root / "uploads",
            registry_path=root / "documents.json",
            persist_dir=root / "chroma",
        )
        service._get_vectorstore = lambda: _FakeVectorStore()  # type: ignore[method-assign]

        service.ingest_upload("handbook.md", b"# Handbook\nWorkplace policy.")
        documents = service.list_documents()

        assert len(documents) == 1
        assert documents[0]["file_name"] == "handbook.md"
        assert documents[0]["source_type"] == "md"


def test_unsupported_type_raises() -> None:
    service = DocumentService()
    try:
        service.ingest_upload("policy.xlsx", b"not-supported")
    except NotImplementedError as exc:
        assert "xlsx" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for unsupported files")


def test_document_service_defaults_to_llama_company_index_dir() -> None:
    service = DocumentService()

    assert service.persist_dir == settings.llama_company_chroma_persist_dir


if __name__ == "__main__":
    test_txt_ingest_persists_metadata()
    test_md_ingest_and_list_documents()
    test_unsupported_type_raises()
    test_document_service_defaults_to_llama_company_index_dir()
    print("document service ok")
