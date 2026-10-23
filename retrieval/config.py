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
    query_embedding_device: str
    embedding_batch_size: int
    ollama_host: str
    search_mode: str
    query_glossary: bool
    reranker_model: str
    use_reranker: bool


def get_settings() -> Settings:
    env = {**_load_dotenv(ROOT / ".env"), **os.environ}
    return Settings(
        chunks_path=ROOT / env.get("CHUNKS_PATH", "data/processed/chunks.jsonl"),
        qdrant_path=ROOT / env.get("QDRANT_PATH", "data/index/qdrant"),
        collection=env.get("QDRANT_COLLECTION", "regulations"),
        embedding_model=env.get("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_device=env.get("EMBEDDING_DEVICE", "auto"),
        # 질문 임베딩은 CPU 로 — GPU 8GB 에 LLM · 리랭커 · 임베딩이 다 올라가면 Windows 가 VRAM 을 시스템 RAM 으로 넘겨
        # LLM 생성이 56 → 6~23 tok/s 로 떨어졌다 (W3 실측). CPU 질문 임베딩은 0.2~0.5초, GPU 1.1GB 확보.
        query_embedding_device=env.get("QUERY_EMBEDDING_DEVICE", "cpu"),
        embedding_batch_size=int(env.get("EMBEDDING_BATCH_SIZE", "16")),
        ollama_host=env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        search_mode=env.get("SEARCH_MODE", "hybrid"),
        query_glossary=env.get("QUERY_GLOSSARY", "on").lower() in ("1", "on", "true", "yes"),
        reranker_model=env.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
        use_reranker=env.get("RERANKER", "on").lower() in ("1", "on", "true", "yes"),
    )
