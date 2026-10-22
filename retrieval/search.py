"""규정 검색: dense(BGE-M3) · bm25(Kiwi) · hybrid(RRF 결합).

    python -m retrieval.search "서울 출장 숙박비 상한" [-k 5] [--mode hybrid] [--org 한국환경공단]
                               [--category 여비] [--section 본칙] [--doc 여비규정] [--full]

Hit.score 는 모드와 관계없이 항상 dense 코사인 유사도다 (검색 단계 거절 기준 MIN_SCORE 가 같은 척도를 쓰도록).
hybrid 순위는 Hit.rrf 로 매긴다.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from .config import ROOT, get_settings
from .embedder import Embedder
from .index import load_chunks, open_client, point_id
from .refs import chunk_ref

FILTER_FIELDS = {"org": "org", "category": "category", "section": "section", "doc": "doc_title"}
MODES = ("dense", "bm25", "hybrid")
POOL = 30        # 각 검색기에서 가져와 합칠 후보 수
RRF_K = 60       # Reciprocal Rank Fusion 상수 (표준값)


@dataclass
class Hit:
    score: float          # dense 코사인 유사도
    chunk: dict
    rrf: float = 0.0      # hybrid 결합 점수
    sources: str = ""     # "dense" · "bm25" · "dense+bm25"

    @property
    def ref(self) -> str:
        return self.chunk.get("ref", "")


def build_filter(**conditions: str | None):
    from qdrant_client import models

    must = [models.FieldCondition(key=FILTER_FIELDS[k], match=models.MatchValue(value=v))
            for k, v in conditions.items() if v]
    return models.Filter(must=must) if must else None


class Searcher:
    """프로세스당 한 번 모델 · 인덱스를 열고 재사용한다 (Qdrant 로컬 모드는 동시에 한 프로세스만 열 수 있음)."""

    def __init__(self, settings=None, mode: str | None = None):
        self.settings = settings or get_settings()
        s = self.settings
        self.mode = mode or s.search_mode
        if self.mode not in MODES:
            raise ValueError(f"mode 는 {MODES} 중 하나: {self.mode}")
        self.embedder = Embedder(s.embedding_model, s.embedding_device, s.embedding_batch_size)
        self.client = open_client(s.qdrant_path)
        if not self.client.collection_exists(s.collection):
            raise RuntimeError("인덱스가 없습니다. 먼저 `python -m retrieval.index` 를 실행하세요.")
        self._bm25 = None

    @property
    def bm25(self):
        if self._bm25 is None:
            from .bm25 import BM25Index

            self._bm25 = BM25Index.build(load_chunks(self.settings.chunks_path), ROOT / "data/index/bm25.pkl")
        return self._bm25

    def warmup(self) -> None:
        """모델 로드 + 첫 GPU 연산(커널 초기화, 약 10초+)을 미리 끝내 첫 질문의 지연이 튀지 않게 한다."""
        self.embedder.embed_query("워밍업")
        if self.mode != "dense":
            self.bm25.search("워밍업", 1)  # Kiwi 로드 · BM25 인덱스 구축(또는 캐시 로드)

    def _dense(self, vector: list[float], k: int, filters: dict) -> list[Hit]:
        result = self.client.query_points(
            self.settings.collection, query=vector, limit=k,
            query_filter=build_filter(**filters), with_payload=True,
        )
        return [Hit(p.score, p.payload, sources="dense") for p in result.points]

    def _bm25_hits(self, query: str, vector: list[float], k: int, filters: dict) -> list[Hit]:
        wanted = {FILTER_FIELDS[name]: v for name, v in filters.items() if v}
        keep = (lambda c: all(c.get(f) == v for f, v in wanted.items())) if wanted else None
        ranked = self.bm25.search(query, k, keep=keep)
        if not ranked:
            return []
        chunks = [self.bm25.chunks[i] for i, _ in ranked]
        points = self.client.retrieve(self.settings.collection, [point_id(c["chunk_id"]) for c in chunks],
                                      with_vectors=True, with_payload=False)
        vectors = {str(p.id): p.vector for p in points}
        hits = []
        for c in chunks:
            vec = vectors.get(point_id(c["chunk_id"]))
            cosine = sum(a * b for a, b in zip(vector, vec)) if vec else 0.0  # 정규화된 벡터 → 내적 = 코사인
            hits.append(Hit(cosine, {**c, "ref": chunk_ref(c)}, sources="bm25"))
        return hits

    def search(self, query: str, k: int = 5, **filters: str | None) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        if self.mode == "dense":
            return self._dense(vector, k, filters)
        if self.mode == "bm25":
            return self._bm25_hits(query, vector, k, filters)
        return rrf_merge(self._dense(vector, POOL, filters), self._bm25_hits(query, vector, POOL, filters))[:k]

    def close(self) -> None:
        self.client.close()


def rrf_merge(*rankings: list[Hit], k: int = RRF_K) -> list[Hit]:
    """여러 검색 결과의 순위를 Reciprocal Rank Fusion 으로 합친다: score = Σ 1 / (k + rank)."""
    merged: dict[str, Hit] = {}
    for hits in rankings:
        for rank, h in enumerate(hits, start=1):
            key = h.chunk["chunk_id"]
            if key not in merged:
                merged[key] = Hit(h.score, h.chunk, 0.0, h.sources)
            elif h.sources not in merged[key].sources:
                merged[key].sources += "+" + h.sources
            merged[key].rrf += 1.0 / (k + rank)
    return sorted(merged.values(), key=lambda h: -h.rrf)


def format_hit(rank: int, hit: Hit, full: bool = False) -> str:
    c = hit.chunk
    title = f"({c['article_title']})" if c.get("article_title") else ""
    part = f" [{c['part'] + 1}/{c['total_parts']}]" if c.get("total_parts", 1) > 1 else ""
    via = f" [{hit.sources}]" if hit.sources and hit.sources != "dense" else ""
    head = (f"{rank:>2}. {hit.score:.3f}  {hit.ref}{title}{part}{via}"
            f"  · {c.get('org', '')} · {c.get('section', '')} · 시행 {c.get('effective_date', '')}")
    body = c["text"] if full else c["text"].replace("\n", " / ")[:220] + ("…" if len(c["text"]) > 220 else "")
    indent = "      "
    return head + "\n" + "\n".join(indent + line for line in body.splitlines())


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="규정 벡터 검색")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--org")
    parser.add_argument("--category")
    parser.add_argument("--section", choices=["본칙", "부칙", "별표", "전문"])
    parser.add_argument("--doc", help="규정명 (sources.csv 의 title, 예: 여비규정 별표1)")
    parser.add_argument("--full", action="store_true", help="본문 전체 출력")
    parser.add_argument("--mode", choices=MODES, help="검색 방식 (기본: .env 의 SEARCH_MODE)")
    args = parser.parse_args(argv)

    searcher = Searcher(mode=args.mode)
    try:
        searcher.warmup()  # 모델 로드 · GPU 초기화를 검색 시간과 분리해서 잰다
        started = time.perf_counter()
        hits = searcher.search(args.query, args.k, org=args.org, category=args.category,
                               section=args.section, doc=args.doc)
        elapsed = (time.perf_counter() - started) * 1000
    finally:
        searcher.close()

    print(f"Q: {args.query}  ({len(hits)}건, {elapsed:.0f}ms)\n")
    for i, hit in enumerate(hits, start=1):
        print(format_hit(i, hit, args.full))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
