"""API schemas for managed law text documents."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class LawDocumentRecord(BaseModel):
    file_name: str
    source_type: str = "txt"
    size_bytes: int = Field(..., ge=0)
    updated_at: str
    rebuild_required: bool = False


class LawDocumentListResponse(BaseModel):
    documents: List[LawDocumentRecord]


class LawDocumentUploadResponse(LawDocumentRecord):
    pass


class LawIndexRebuildResponse(BaseModel):
    message: str
    indexed_document_count: int = Field(..., ge=0)
    indexed_node_count: int = Field(..., ge=0)
    rebuild_required: bool = False
