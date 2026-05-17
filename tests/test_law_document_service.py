"""Law document management checks without loading live models."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.law_document_service import LawDocumentService  # noqa: E402


def test_upload_law_txt_persists_file_and_metadata() -> None:
    with TemporaryDirectory() as temp_dir:
        service = LawDocumentService(data_dir=temp_dir)

        record = service.ingest_upload(
            "劳动法补充条文.txt",
            "第一条 测试法律条文。".encode("utf-8"),
        )

        assert record["file_name"] == "劳动法补充条文.txt"
        assert record["source_type"] == "txt"
        assert record["size_bytes"] > 0
        assert record["rebuild_required"] is True
        assert (Path(temp_dir) / "劳动法补充条文.txt").read_text(encoding="utf-8") == "第一条 测试法律条文。"


def test_law_documents_only_accept_txt() -> None:
    service = LawDocumentService(data_dir="unused")

    try:
        service.ingest_upload("劳动法.pdf", b"not-supported")
    except NotImplementedError as exc:
        assert "txt" in str(exc)
    else:
        raise AssertionError("Expected NotImplementedError for non-txt law documents")


def test_list_law_documents_returns_txt_metadata() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "a.txt").write_text("A", encoding="utf-8")
        (root / "b.md").write_text("B", encoding="utf-8")
        service = LawDocumentService(data_dir=root)

        documents = service.list_documents()

        assert [doc["file_name"] for doc in documents] == ["a.txt"]
        assert documents[0]["source_type"] == "txt"
        assert documents[0]["size_bytes"] == 1
        assert documents[0]["updated_at"]


def test_rebuild_index_invokes_pipeline_and_resets_runtime_cache() -> None:
    calls = []

    class FakePipeline:
        def __init__(self, *, rebuild_index: bool, verbose: bool) -> None:
            calls.append({"rebuild_index": rebuild_index, "verbose": verbose})
            self.nodes = ["node-1", "node-2"]

    class FakeRAGService:
        def __init__(self) -> None:
            self.reset_called = False

        def reset_pipeline(self) -> None:
            self.reset_called = True

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        (root / "law.txt").write_text("第一条 测试。", encoding="utf-8")
        rag_service = FakeRAGService()
        service = LawDocumentService(
            data_dir=root,
            pipeline_factory=FakePipeline,
            rag_service=rag_service,
        )

        result = service.rebuild_index()

        assert calls == [{"rebuild_index": True, "verbose": False}]
        assert rag_service.reset_called is True
        assert result["indexed_document_count"] == 1
        assert result["indexed_node_count"] == 2
        assert result["rebuild_required"] is False
