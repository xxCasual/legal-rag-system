"""
Central configuration for the enterprise legal RAG application.

This module keeps project paths, model defaults, and runtime feature flags in
one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    return Path(value).expanduser().resolve()


def _choice_from_env(name: str, default: str, allowed: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"{name} must be one of: {choices}")
    return value


def _float_from_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = _path_from_env("LEGAL_RAG_DATA_DIR", PROJECT_ROOT / "data")
    chroma_persist_dir: Path = _path_from_env(
        "LEGAL_RAG_CHROMA_DIR", PROJECT_ROOT / "chroma_db"
    )
    llama_law_chroma_persist_dir: Path = _path_from_env(
        "LEGAL_RAG_LLAMA_LAW_CHROMA_DIR", PROJECT_ROOT / "chroma_llama_law"
    )
    embedding_model: str = os.getenv(
        "LEGAL_RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    )
    reranker_model: str = os.getenv(
        "LEGAL_RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
    )
    crag_mode: str = _choice_from_env(
        "LEGAL_RAG_CRAG_MODE",
        "reranker",
        {"llm", "reranker", "off"},
    )
    llm_model: str = os.getenv("LEGAL_RAG_LLM_MODEL", "deepseek-chat")
    llm_timeout_seconds: float = _float_from_env("LEGAL_RAG_LLM_TIMEOUT_SECONDS", 30.0)
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    storage_dir: Path = _path_from_env("LEGAL_RAG_STORAGE_DIR", PROJECT_ROOT / "storage")
    uploads_dir: Path = _path_from_env(
        "LEGAL_RAG_UPLOADS_DIR", PROJECT_ROOT / "storage" / "uploads"
    )
    document_registry_path: Path = _path_from_env(
        "LEGAL_RAG_DOCUMENT_REGISTRY",
        PROJECT_ROOT / "storage" / "documents.json",
    )
    pending_reviews_path: Path = _path_from_env(
        "LEGAL_RAG_PENDING_REVIEWS",
        PROJECT_ROOT / "storage" / "pending_reviews.json",
    )
    company_chroma_persist_dir: Path = _path_from_env(
        "LEGAL_RAG_COMPANY_CHROMA_DIR",
        PROJECT_ROOT / "chroma_company_docs",
    )
    llama_company_chroma_persist_dir: Path = _path_from_env(
        "LEGAL_RAG_LLAMA_COMPANY_CHROMA_DIR",
        PROJECT_ROOT / "chroma_llama_company",
    )


settings = Settings()
