"""Configuration checks for the LlamaIndex refactor."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402


def test_llama_index_chroma_paths_default_to_v2_directories() -> None:
    assert settings.llama_law_chroma_persist_dir == ROOT / "chroma_llama_law"
    assert settings.llama_company_chroma_persist_dir == ROOT / "chroma_llama_company"


if __name__ == "__main__":
    test_llama_index_chroma_paths_default_to_v2_directories()
    print("config ok")
