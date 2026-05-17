"""Manage uploaded law text documents and legal index rebuilds."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List

from app.core.config import settings


PipelineFactory = Callable[..., Any]


class LawDocumentService:
    """Persist law text files and rebuild the dedicated legal RAG index."""

    SUPPORTED_TYPE = "txt"

    def __init__(
        self,
        data_dir: Path | str | None = None,
        state_path: Path | str | None = None,
        pipeline_factory: PipelineFactory | None = None,
        rag_service: object | None = None,
    ) -> None:
        self.data_dir = Path(data_dir or settings.data_dir)
        self.state_path = Path(
            state_path or settings.storage_dir / "law_documents_state.json"
        )
        self._pipeline_factory = pipeline_factory
        self._rag_service = rag_service
        self._lock = Lock()

    def ingest_upload(self, file_name: str, content: bytes) -> Dict[str, Any]:
        source_type = self._detect_source_type(file_name)
        safe_name = Path(file_name).name or f"uploaded_law.{source_type}"
        self._validate_text(content)

        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            saved_path = self.data_dir / safe_name
            saved_path.write_bytes(content)
            self._write_state({"rebuild_required": True})
            return {
                **self._record_for_path(saved_path),
                "rebuild_required": True,
            }

    def list_documents(self) -> List[Dict[str, Any]]:
        rebuild_required = self._rebuild_required()
        if not self.data_dir.exists():
            return []
        return [
            {**self._record_for_path(path), "rebuild_required": rebuild_required}
            for path in sorted(self.data_dir.glob("*.txt"), key=lambda item: item.name)
            if path.is_file()
        ]

    def rebuild_index(self) -> Dict[str, Any]:
        pipeline_factory = self._pipeline_factory or self._default_pipeline_factory
        with self._lock:
            pipeline = pipeline_factory(rebuild_index=True, verbose=False)
            self._reset_rag_service()
            self._write_state({"rebuild_required": False})

        return {
            "message": "法律索引已重建。",
            "indexed_document_count": len(self.list_documents()),
            "indexed_node_count": len(getattr(pipeline, "nodes", []) or []),
            "rebuild_required": False,
        }

    @classmethod
    def _detect_source_type(cls, file_name: str) -> str:
        source_type = Path(file_name).suffix.lower().lstrip(".")
        if source_type != cls.SUPPORTED_TYPE:
            raise NotImplementedError("Law documents only support .txt files")
        return source_type

    def _validate_text(self, content: bytes) -> None:
        content.decode("utf-8-sig")

    def _record_for_path(self, path: Path) -> Dict[str, Any]:
        stat = path.stat()
        return {
            "file_name": path.name,
            "source_type": self.SUPPORTED_TYPE,
            "size_bytes": stat.st_size,
            "updated_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            "rebuild_required": False,
        }

    def _rebuild_required(self) -> bool:
        if not self.state_path.exists():
            return False
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        return bool(data.get("rebuild_required")) if isinstance(data, dict) else False

    def _write_state(self, data: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _default_pipeline_factory(self, **kwargs: Any) -> Any:
        from app.rag import LlamaIndexLegalRAGPipeline

        return LlamaIndexLegalRAGPipeline(**kwargs)

    def _reset_rag_service(self) -> None:
        rag_service = self._rag_service
        if rag_service is None:
            from app.services.rag_service import rag_service as default_rag_service

            rag_service = default_rag_service
        reset = getattr(rag_service, "reset_pipeline", None)
        if callable(reset):
            reset()


law_document_service = LawDocumentService()
