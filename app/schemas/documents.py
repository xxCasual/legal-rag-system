"""API schemas for uploaded company documents."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    doc_id: str
    file_name: str
    source_type: str
    chunk_count: int = Field(..., ge=0)
    created_at: str


class DocumentUploadResponse(DocumentRecord):
    pass


class DocumentListResponse(BaseModel):
    documents: List[DocumentRecord]
