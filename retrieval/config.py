"""설정: 환경변수 > .env > 기본값. (python-dotenv 없이 단순 파서 사용)"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class Settings:
    chunks_path: Path
    qdrant_path: Path
    collection: str
    embedding_model: str
    embedding_device: str
    embedding_batch_size: int
    ollama_host: str
    search_mode: str


def get_settings() -> Settings:
    env = {**_load_dotenv(ROOT / ".env"), **os.environ}
    return Settings(
        chunks_path=ROOT / env.get("CHUNKS_PATH", "data/processed/chunks.jsonl"),
        qdrant_path=ROOT / env.get("QDRANT_PATH", "data/index/qdrant"),
        collection=env.get("QDRANT_COLLECTION", "regulations"),
        embedding_model=env.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_device=env.get("EMBEDDING_DEVICE", "auto"),
        embedding_batch_size=int(env.get("EMBEDDING_BATCH_SIZE", "16")),
        ollama_host=env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        search_mode=env.get("SEARCH_MODE", "hybrid"),
    )
