"""Upload, parse, split, and index company policy documents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from app.core.config import settings


@dataclass
class PolicyChunk:
    """Small document shape returned by company policy search."""

    page_content: str
    metadata: Dict[str, Any]


class DocumentService:
    """Manage enterprise documents in a dedicated LlamaIndex Chroma index."""

    SUPPORTED_TYPES = {"txt", "md", "pdf", "docx"}
    COMPANY_COLLECTION = "company_policy_docs"

    def __init__(
        self,
        uploads_dir: Path | str | None = None,
        registry_path: Path | str | None = None,
        persist_dir: Path | str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self.uploads_dir = Path(uploads_dir or settings.uploads_dir)
        self.registry_path = Path(registry_path or settings.document_registry_path)
        self.persist_dir = Path(persist_dir or settings.llama_company_chroma_persist_dir)
        self.embedding_model = embedding_model or settings.embedding_model
        self._index: Any | None = None
        self._lock = Lock()

    def ingest_upload(self, file_name: str, content: bytes) -> Dict[str, Any]:
        source_type = self._detect_source_type(file_name)
        text = self._parse_content(source_type, content)
        doc_id = str(uuid4())
        safe_name = Path(file_name).name or f"{doc_id}.{source_type}"

        with self._lock:
            self._ensure_storage()
            saved_path = self.uploads_dir / f"{doc_id}_{safe_name}"
            saved_path.write_bytes(content)

            chunks = self._build_chunks(doc_id, safe_name, source_type, text)
            self._index_chunks(chunks)

            record = {
                "doc_id": doc_id,
                "file_name": safe_name,
                "source_type": source_type,
                "chunk_count": len(chunks),
                "created_at": datetime.now(UTC).isoformat(),
            }
            records = self._load_registry()
            records.append(record)
            self._write_registry(records)
        return record

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load_registry()

    def search(self, query: str, k: int = 4) -> List[PolicyChunk]:
        if not self.persist_dir.exists() or not any(self.persist_dir.iterdir()):
            return []
        index = self._get_vectorstore()
        if hasattr(index, "similarity_search"):
            return index.similarity_search(query, k=k)
        retriever = index.as_retriever(similarity_top_k=k)
        return [self._chunk_from_node(item) for item in retriever.retrieve(query)]

    @classmethod
    def _detect_source_type(cls, file_name: str) -> str:
        source_type = Path(file_name).suffix.lower().lstrip(".")
        if source_type not in cls.SUPPORTED_TYPES:
            raise NotImplementedError(f"Unsupported document type: {source_type or 'unknown'}")
        return source_type

    def _parse_content(self, source_type: str, content: bytes) -> str:
        if source_type in {"txt", "md"}:
            return content.decode("utf-8-sig")
        if source_type == "pdf":
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if source_type == "docx":
            from docx import Document as DocxDocument

            document = DocxDocument(BytesIO(content))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        raise NotImplementedError(f"Unsupported document type: {source_type}")

    def _build_chunks(
        self,
        doc_id: str,
        file_name: str,
        source_type: str,
        text: str,
    ) -> List[PolicyChunk]:
        return [
            PolicyChunk(
                page_content=chunk,
                metadata={
                    "doc_id": doc_id,
                    "file_name": file_name,
                    "source_type": source_type,
                    "chunk_id": chunk_id,
                },
            )
            for chunk_id, chunk in enumerate(self._split_text(text))
        ]

    def _split_text(
        self,
        text: str,
        chunk_size: int = 1500,
        chunk_overlap: int = 300,
    ) -> List[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(cleaned):
            end = min(start + chunk_size, len(cleaned))
            chunks.append(cleaned[start:end])
            if end == len(cleaned):
                break
            start = max(end - chunk_overlap, start + 1)
        return chunks

    def _index_chunks(self, chunks: Iterable[PolicyChunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return
        index = self._get_vectorstore()
        if hasattr(index, "add_documents"):
            index.add_documents(chunk_list)
            return
        index.insert_nodes([self._node_from_chunk(chunk) for chunk in chunk_list])

    def _get_vectorstore(self) -> Any:
        if self._index is None:
            import chromadb
            from llama_index.core import Settings, StorageContext, VectorStoreIndex
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.vector_stores.chroma import ChromaVectorStore

            Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            collection = client.get_or_create_collection(self.COMPANY_COLLECTION)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            if collection.count() == 0:
                self._index = VectorStoreIndex([], storage_context=storage_context)
            else:
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context,
                )
        return self._index

    def _node_from_chunk(self, chunk: PolicyChunk) -> Any:
        try:
            from llama_index.core.schema import TextNode

            return TextNode(text=chunk.page_content, metadata=chunk.metadata)
        except ImportError:
            return chunk

    def _chunk_from_node(self, item: Any) -> PolicyChunk:
        node = getattr(item, "node", item)
        metadata = dict(getattr(node, "metadata", {}) or {})
        if hasattr(node, "get_content"):
            text = str(node.get_content())
        else:
            text = str(getattr(node, "text", getattr(node, "page_content", "")))
        return PolicyChunk(page_content=text, metadata=metadata)

    def _ensure_storage(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    def _load_registry(self) -> List[Dict[str, Any]]:
        if not self.registry_path.exists():
            return []
        data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _write_registry(self, records: List[Dict[str, Any]]) -> None:
        self.registry_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


document_service = DocumentService()
