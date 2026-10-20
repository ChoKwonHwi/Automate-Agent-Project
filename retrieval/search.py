"""벡터 검색.

    python -m retrieval.search "서울 출장 숙박비 상한" [-k 5] [--org 한국환경공단] [--category 여비]
                               [--section 본칙] [--doc 여비규정] [--full]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from .config import get_settings
from .embedder import Embedder
from .index import open_client

FILTER_FIELDS = {"org": "org", "category": "category", "section": "section", "doc": "doc_title"}


@dataclass
class Hit:
    score: float
    chunk: dict

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

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        s = self.settings
        self.embedder = Embedder(s.embedding_model, s.embedding_device, s.embedding_batch_size)
        self.client = open_client(s.qdrant_path)
        if not self.client.collection_exists(s.collection):
            raise RuntimeError("인덱스가 없습니다. 먼저 `python -m retrieval.index` 를 실행하세요.")

    def warmup(self) -> None:
        """모델 로드 + 첫 GPU 연산(커널 초기화, 약 10초+)을 미리 끝내 첫 질문의 지연이 튀지 않게 한다."""
        self.embedder.embed_query("워밍업")

    def search(self, query: str, k: int = 5, **filters: str | None) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        result = self.client.query_points(
            self.settings.collection, query=vector, limit=k,
            query_filter=build_filter(**filters), with_payload=True,
        )
        return [Hit(p.score, p.payload) for p in result.points]

    def close(self) -> None:
        self.client.close()


def format_hit(rank: int, hit: Hit, full: bool = False) -> str:
    c = hit.chunk
    title = f"({c['article_title']})" if c.get("article_title") else ""
    part = f" [{c['part'] + 1}/{c['total_parts']}]" if c.get("total_parts", 1) > 1 else ""
    head = (f"{rank:>2}. {hit.score:.3f}  {hit.ref}{title}{part}"
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
    args = parser.parse_args(argv)

    searcher = Searcher()
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
