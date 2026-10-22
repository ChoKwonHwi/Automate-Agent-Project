"""BM25 키워드 검색 (Kiwi 형태소 분석).

dense(BGE-M3)는 의미가 비슷한 문장을 잘 찾지만, "연봉월액" · "적격심사" 같은 규정 용어를 정확히 짚는 데는
키워드 검색이 강하다. 두 결과는 retrieval.hybrid 에서 RRF 로 합친다.

토큰: 명사 · 동사/형용사 어간 · 숫자 · 외국어 + 연속 명사를 이어 붙인 복합어("강사"+"료" → "강사료").
인덱스는 chunks.jsonl 내용 해시가 같으면 data/index/bm25.pkl 캐시를 재사용한다.
"""
from __future__ import annotations

import hashlib
import math
import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

_KEEP_TAGS = {"NNG", "NNP", "NR", "SN", "SL", "SH", "XR", "VV", "VA", "NNB"}
_STOP = {"수", "것", "등", "때", "바", "중", "경우", "하", "되", "있", "없", "이", "그", "및", "해당"}
CACHE_VERSION = 1


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi

    return Kiwi()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    run: list[str] = []  # 연속 명사
    for t in _kiwi().tokenize(text or ""):
        tag = t.tag.split("-")[0]
        if tag in ("NNG", "NNP"):
            run.append(t.form)
        else:
            if len(run) >= 2:
                tokens.append("".join(run))
            run = []
        if tag in _KEEP_TAGS and t.form not in _STOP:
            tokens.append(t.form)
    if len(run) >= 2:
        tokens.append("".join(run))
    return tokens


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(toks) for toks in docs_tokens]
        self.len = [len(toks) for toks in docs_tokens]
        self.avgdl = sum(self.len) / len(self.len) if self.len else 0.0
        df = Counter(term for tf in self.tf for term in tf)
        n = len(docs_tokens)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * len(self.tf)
        terms = [t for t in dict.fromkeys(query_tokens) if t in self.idf]
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avgdl)
            s = 0.0
            for term in terms:
                f = tf.get(term)
                if f:
                    s += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            out[i] = s
        return out


class BM25Index:
    """chunks 와 같은 순서의 BM25 인덱스."""

    def __init__(self, chunks: list[dict], bm25: BM25):
        self.chunks = chunks
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks: list[dict], cache_path: Path | None = None) -> "BM25Index":
        digest = hashlib.sha1("\n".join(c["chunk_id"] + c["embed_text"] for c in chunks).encode("utf-8")).hexdigest()
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("version") == CACHE_VERSION and cached.get("digest") == digest:
                    return cls(chunks, cached["bm25"])
            except Exception:
                pass
        bm25 = BM25([tokenize(c["embed_text"]) for c in chunks])
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump({"version": CACHE_VERSION, "digest": digest, "bm25": bm25}, f)
        return cls(chunks, bm25)

    def search(self, query: str, k: int = 30, keep=None) -> list[tuple[int, float]]:
        """[(chunk 인덱스, 점수)] 점수 내림차순. keep(chunk) → bool 로 필터."""
        scores = self.bm25.scores(tokenize(query))
        ranked = sorted((i for i, s in enumerate(scores) if s > 0 and (keep is None or keep(self.chunks[i]))),
                        key=lambda i: -scores[i])
        return [(i, scores[i]) for i in ranked[:k]]
