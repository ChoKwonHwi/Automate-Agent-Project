"""청크를 Qdrant(로컬 모드)에 적재한다.

    python -m retrieval.index [--rebuild] [--keep-ollama]

- 포인트 ID 는 chunk_id 의 UUID5 → 같은 청크는 덮어쓰기 (멱등)
- 본문 해시가 같으면 재임베딩하지 않는다 (증분 인덱싱)
- chunks.jsonl 에서 사라진 청크는 인덱스에서도 지운다
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from .config import ROOT, get_settings
from .refs import chunk_ref

EmbedFn = Callable[[list[str]], list[list[float]]]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def content_hash(chunk: dict, model_name: str) -> str:
    return hashlib.sha1(f"{model_name}\n{chunk['embed_text']}".encode("utf-8")).hexdigest()


def load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def open_client(path: Path):
    from qdrant_client import QdrantClient

    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def ensure_collection(client, name: str, dim: int, rebuild: bool = False) -> None:
    from qdrant_client import models

    if rebuild and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))


def existing_hashes(client, name: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(name, limit=512, offset=offset, with_payload=["content_hash"], with_vectors=False)
        for p in points:
            hashes[str(p.id)] = (p.payload or {}).get("content_hash", "")
        if offset is None:
            return hashes


def index_chunks(client, collection: str, chunks: list[dict], embed: EmbedFn, model_name: str,
                 batch_size: int = 64, log: Callable[[str], None] = print) -> dict[str, int]:
    from qdrant_client import models

    current = {point_id(c["chunk_id"]): c for c in chunks}
    stored = existing_hashes(client, collection)

    stale = [pid for pid in stored if pid not in current]
    if stale:
        client.delete(collection, points_selector=models.PointIdsList(points=stale))

    todo = [(pid, c, content_hash(c, model_name)) for pid, c in current.items()]
    todo = [(pid, c, h) for pid, c, h in todo if stored.get(pid) != h]
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        vectors = embed([c["embed_text"] for _, c, _ in batch])
        client.upsert(collection, points=[
            models.PointStruct(id=pid, vector=vec, payload={**c, "ref": chunk_ref(c), "content_hash": h})
            for (pid, c, h), vec in zip(batch, vectors)
        ])
        log(f"  임베딩 {min(start + batch_size, len(todo))}/{len(todo)}")
    return {"total": len(current), "embedded": len(todo), "deleted": len(stale), "unchanged": len(current) - len(todo)}


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="청크를 벡터 인덱스에 적재")
    parser.add_argument("--rebuild", action="store_true", help="컬렉션을 지우고 처음부터 다시 만든다")
    parser.add_argument("--keep-ollama", action="store_true", help="Ollama 모델을 내리지 않는다")
    args = parser.parse_args(argv)

    from .embedder import Embedder, unload_ollama_models

    s = get_settings()
    if not s.chunks_path.exists():
        print(f"{s.chunks_path} 가 없습니다. 먼저 `python -m ingestion.run` 을 실행하세요.")
        return 1
    chunks = load_chunks(s.chunks_path)
    if not args.keep_ollama and (names := unload_ollama_models(s.ollama_host)):
        print(f"VRAM 확보: Ollama 모델 언로드 {names}")

    embedder = Embedder(s.embedding_model, s.embedding_device, s.embedding_batch_size)
    started = time.perf_counter()
    print(f"임베딩 모델 로드: {s.embedding_model} ...", flush=True)
    dim = embedder.dim
    print(f"  device={embedder.model.device}, dim={dim}")

    client = open_client(s.qdrant_path)
    try:
        ensure_collection(client, s.collection, dim, rebuild=args.rebuild)
        stats = index_chunks(client, s.collection, chunks, embedder.embed, s.embedding_model)
    finally:
        client.close()
    elapsed = time.perf_counter() - started
    print(f"완료 ({elapsed:.1f}초): 전체 {stats['total']} · 임베딩 {stats['embedded']} · "
          f"변경없음 {stats['unchanged']} · 삭제 {stats['deleted']}")
    print(f"→ {s.qdrant_path.relative_to(ROOT).as_posix()} / {s.collection}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
