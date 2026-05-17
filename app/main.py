"""FastAPI entrypoint for the enterprise legal RAG platform."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent import run_agent_chat
from app.agent.tools import review_labor_contract
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ContractReviewRequest,
    ContractReviewResponse,
    DocumentListResponse,
    DocumentRecord,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    LawDocumentListResponse,
    LawDocumentRecord,
    LawDocumentUploadResponse,
    LawIndexRebuildResponse,
    PendingReviewListResponse,
    PendingReviewRecord,
    ReviewDecisionResponse,
)
from app.services.document_service import document_service
from app.services.law_document_service import law_document_service
from app.services.review_service import (
    ReviewNotFoundError,
    ReviewStateError,
    review_service,
)


STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Enterprise Legal RAG Platform",
    description="V5-first Chinese labor compliance RAG assistant.",
    version="0.2.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_, exc: Exception) -> JSONResponse:
    payload = ErrorResponse(error="internal_error", detail=str(exc))
    content = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    return JSONResponse(status_code=500, content=content)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    result = run_agent_chat(request.query)
    return ChatResponse(**result)


@app.post("/api/review/contract", response_model=ContractReviewResponse)
async def review_contract(request: ContractReviewRequest) -> ContractReviewResponse:
    result = review_labor_contract(request.contract_text, include_evidence=True)
    contract_review = result.get("contract_review") or {}
    return ContractReviewResponse(**contract_review)


@app.get("/api/reviews/pending", response_model=PendingReviewListResponse)
async def list_pending_reviews() -> PendingReviewListResponse:
    records = [
        PendingReviewRecord(**record) for record in review_service.list_pending_reviews()
    ]
    return PendingReviewListResponse(reviews=records)


@app.post("/api/reviews/{review_id}/approve", response_model=ReviewDecisionResponse)
async def approve_review(review_id: str) -> ReviewDecisionResponse:
    try:
        result = review_service.approve_review(review_id)
    except ReviewNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Review not found: {review_id}") from exc
    except ReviewStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ReviewDecisionResponse(**result)


@app.post("/api/reviews/{review_id}/reject", response_model=ReviewDecisionResponse)
async def reject_review(review_id: str) -> ReviewDecisionResponse:
    try:
        result = review_service.reject_review(review_id)
    except ReviewNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Review not found: {review_id}") from exc
    except ReviewStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ReviewDecisionResponse(**result)


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    file_name = file.filename or "uploaded_document"
    content = await file.read()
    try:
        record = document_service.ingest_upload(file_name, content)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return DocumentUploadResponse(**record)


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    records = [DocumentRecord(**record) for record in document_service.list_documents()]
    return DocumentListResponse(documents=records)


@app.get("/api/law-documents", response_model=LawDocumentListResponse)
async def list_law_documents() -> LawDocumentListResponse:
    records = [
        LawDocumentRecord(**record) for record in law_document_service.list_documents()
    ]
    return LawDocumentListResponse(documents=records)


@app.post("/api/law-documents/upload", response_model=LawDocumentUploadResponse)
async def upload_law_document(file: UploadFile = File(...)) -> LawDocumentUploadResponse:
    file_name = file.filename or "uploaded_law.txt"
    content = await file.read()
    try:
        record = law_document_service.ingest_upload(file_name, content)
    except (NotImplementedError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LawDocumentUploadResponse(**record)


@app.post("/api/law-documents/rebuild-index", response_model=LawIndexRebuildResponse)
async def rebuild_law_index() -> LawIndexRebuildResponse:
    result = law_document_service.rebuild_index()
    return LawIndexRebuildResponse(**result)
