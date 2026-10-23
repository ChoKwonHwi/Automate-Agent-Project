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
# 쿼리 재작성 기준: 상위 결과의 최고 dense 점수가 이보다 낮을 때만 재작성 (W3 측정: Golden Set 하위 10% = 0.622,
# 검색 실패한 일상어 질문 0.580) → 평소 질문은 재작성 없이 처리해 지연을 아낀다
REWRITE_BELOW = 0.62
RERANK_TOP = 20  # 리랭커로 다시 줄 세울 후보 수 (GPU 0.4초)


@dataclass
class Hit:
    score: float          # dense 코사인 유사도
    chunk: dict
    rrf: float = 0.0      # hybrid 결합 점수
    sources: str = ""     # "dense" · "bm25" · "dense+bm25"
    rerank: float | None = None  # 리랭커 관련도 0~1 (리랭커를 켰을 때)

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
        self.embedder = Embedder(s.embedding_model, s.query_embedding_device, s.embedding_batch_size)
        self.client = open_client(s.qdrant_path)
        if not self.client.collection_exists(s.collection):
            raise RuntimeError("인덱스가 없습니다. 먼저 `python -m retrieval.index` 를 실행하세요.")
        self._bm25 = None
        self.use_glossary = s.query_glossary  # 일상어 → 규정 용어 사전 (hybrid 모드, 지연 ~0ms)
        self.last_expansion: list[str] = []
        self.rewriter = None          # enable_rewrite() 로 켠다 (hybrid 모드에서만 동작, 기본 꺼짐 — W3 실험 결과)
        self.reranker = None          # enable_rerank() 로 켠다 (설정 RERANKER=on 이면 hybrid 에서 자동)
        self.rewrite_below = REWRITE_BELOW
        self.last_rewrite = None      # 마지막 검색의 재작성 결과 (평가 · 디버깅용)
        if s.use_reranker and self.mode == "hybrid":
            self.enable_rerank()

    def enable_rerank(self, reranker=None) -> None:
        from .reranker import Reranker

        self.reranker = reranker or Reranker(self.settings.reranker_model, "auto")  # GPU (CPU 는 질문당 83초)

    def _rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        """RRF 상위 RERANK_TOP 개를 리랭커 관련도로 다시 줄 세운다. 그 밖의 후보는 버린다."""
        pool = hits[:max(RERANK_TOP, k)]
        for h, score in zip(pool, self.reranker.score(query, [h.chunk["embed_text"] for h in pool])):
            h.rerank = score
        return sorted(pool, key=lambda h: -h.rerank)[:k]

    def enable_rewrite(self, llm, below: float = REWRITE_BELOW, always: bool = False) -> None:
        from .rewrite import QueryRewriter

        self.rewriter = QueryRewriter(llm)
        self.rewrite_below = 1.01 if always else below

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
        if self.reranker:
            self.reranker.warmup()

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
        self.last_rewrite = None
        self.last_expansion = []
        expanded = None
        if self.use_glossary and self.mode == "hybrid":
            from .glossary import expand

            self.last_expansion = expand(query)
            if self.last_expansion:
                expanded = query + " " + " ".join(self.last_expansion)
        # 원래 질문과 확장 질문을 한 번에 임베딩한다 (질문 임베딩이 CPU 라 호출 횟수가 지연을 좌우)
        vectors = self.embedder.embed([query, expanded] if expanded else [query])
        vector = vectors[0]
        if self.mode == "dense":
            return self._dense(vector, k, filters)
        if self.mode == "bm25":
            return self._bm25_hits(query, vector, k, filters)
        rankings = [self._dense(vector, POOL, filters), self._bm25_hits(query, vector, POOL, filters)]
        score_vectors = [vector]
        if expanded:
            # 원래 질문 검색에 "원래 질문 + 규정 용어" 검색을 더한다 (add).
            # 확장 검색만 쓰면 사전이 빗나갈 때 원래 결과를 통째로 잃는다
            # (W3 일상어 16문항: q081 "교통비"→운임 확장으로 6위 → 10위 밖. add 가 전체 · 일상어 모두 최고 MRR)
            ev = vectors[1]
            score_vectors.append(ev)
            rankings += [self._dense(ev, POOL, filters), self._bm25_hits(expanded, ev, POOL, filters)]
        merged = rrf_merge(*rankings)
        if len(score_vectors) > 1:
            self._rescore(merged[:max(k, 10)], score_vectors)
        best = max((h.score for h in merged[:10]), default=0.0)
        if self.rewriter is None or best >= self.rewrite_below:
            return self._finish(query, merged, k)

        rw = self.rewriter.rewrite(query)
        self.last_rewrite = rw
        if rw.empty:
            return self._finish(query, merged, k)
        for q in rw.queries:
            qv = self.embedder.embed_query(q)
            score_vectors.append(qv)
            rankings += [self._dense(qv, POOL, filters), self._bm25_hits(q, qv, POOL, filters)]
        if rw.hypothetical:  # 가상 조문은 검색에만 쓴다 (근거 · 점수에는 쓰지 않음)
            rankings.append(self._dense(self.embedder.embed_query(rw.hypothetical), POOL, filters))
        merged = rrf_merge(*rankings)[:max(k, RERANK_TOP)]
        self._rescore(merged, score_vectors)
        return self._finish(query, merged, k)

    def _finish(self, query: str, merged: list[Hit], k: int) -> list[Hit]:
        return self._rerank(query, merged, k) if self.reranker else merged[:k]

    def _rescore(self, hits: list[Hit], vectors: list[list[float]]) -> None:
        """Hit.score 를 원래 질문 · 재작성 검색어 벡터와의 코사인 중 큰 값으로 맞춘다 (거절 기준의 척도 통일)."""
        points = self.client.retrieve(self.settings.collection, [point_id(h.chunk["chunk_id"]) for h in hits],
                                      with_vectors=True, with_payload=False)
        stored = {str(p.id): p.vector for p in points}
        for h in hits:
            vec = stored.get(point_id(h.chunk["chunk_id"]))
            if vec:
                h.score = max(sum(a * b for a, b in zip(q, vec)) for q in vectors)

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
    if hit.rerank is not None:
        via += f" rerank {hit.rerank:.3f}"
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
    parser.add_argument("--rewrite", choices=["auto", "always"], help="LLM 쿼리 재작성 (hybrid 모드, 실험용)")
    parser.add_argument("--no-glossary", action="store_true", help="일상어 사전 확장 끄기")
    parser.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                        help="리랭커 재정렬 켜기/끄기 (기본: .env 의 RERANKER)")
    args = parser.parse_args(argv)

    searcher = Searcher(mode=args.mode)
    if args.no_glossary:
        searcher.use_glossary = False
    if args.rerank:
        searcher.enable_rerank()
    elif args.rerank is False:
        searcher.reranker = None
    if args.rewrite:
        from rag.llm import OllamaChat

        searcher.enable_rewrite(OllamaChat(), always=args.rewrite == "always")
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
