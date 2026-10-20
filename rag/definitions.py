"""정의 조항 자동 첨부.

규정은 보통 앞부분 정의 조항(예: 보수규정 제2조)에서 용어를 정의하고 본문에서 그 용어를 쓴다.
정의 조항은 질문과 의미가 덜 비슷해 검색 순위가 낮지만, "연봉월액" 같은 용어를 묻는 질문의 핵심 근거다.
질문에 정의된 용어가 나오면 해당 정의 줄만 작은 근거 블록으로 붙인다.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from retrieval.refs import chunk_ref, normalize_ref

_ITEM_RE = re.compile(r'^\s*\d+\.\s*["“「]([^"”」\n]{1,30})["”」]\s*(?:이)?(?:란|라\s*함은)')
MIN_TERM_LEN = 3      # "보수", "민원" 처럼 짧은 용어는 거의 모든 질문에 걸려 잡음이 된다
MAX_LINES = 3
LINE_CAP = 300


@dataclass(frozen=True)
class Definition:
    term: str
    line: str
    ref: str
    doc_title: str
    org: str
    effective_date: str


def _key(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _variants(term: str) -> set[str]:
    k = _key(term)
    return {k, k.replace("의", "")}  # "연봉의 일할계산" ↔ "연봉 일할계산"


def build_index(chunks: list[dict]) -> dict[str, list[Definition]]:
    """doc_title → 정의 목록"""
    index: dict[str, list[Definition]] = defaultdict(list)
    for c in chunks:
        if c["section"] != "본칙" or "정의" not in c.get("article_title", ""):
            continue
        for line in c["text"].splitlines():
            m = _ITEM_RE.match(line)
            if m and len(_key(m.group(1))) >= MIN_TERM_LEN:
                index[c["doc_title"]].append(Definition(m.group(1).strip(), line.strip()[:LINE_CAP], chunk_ref(c),
                                                        c["doc_title"], c.get("org", ""), c.get("effective_date", "")))
    return dict(index)


def matching_definitions(question: str, doc_titles: list[str], index: dict[str, list[Definition]],
                         exclude_refs: set[str] = frozenset(), max_lines: int = MAX_LINES) -> list[Definition]:
    q_variants = {_key(question), _key(question).replace("의", "")}
    found = []
    for title in dict.fromkeys(doc_titles):
        for d in index.get(title, []):
            if normalize_ref(d.ref) in exclude_refs:
                continue
            if any(v in q for v in _variants(d.term) for q in q_variants):
                found.append(d)
    found.sort(key=lambda d: -len(_key(d.term)))  # 긴(구체적인) 용어 우선
    return found[:max_lines]


def definition_blocks(defs: list[Definition]) -> list[dict]:
    """같은 정의 조항의 줄들을 하나의 근거 블록으로 묶는다."""
    grouped: dict[str, dict] = {}
    for d in defs:
        b = grouped.setdefault(d.ref, {"ref": d.ref, "doc_title": d.doc_title, "org": d.org,
                                       "effective_date": d.effective_date, "score": 0.0, "lines": []})
        b["lines"].append(d.line)
    return [{**{k: v for k, v in b.items() if k != "lines"},
             "text": f"{b['ref']} 중 질문 관련 용어 정의\n" + "\n".join(b["lines"]), "kind": "definition"}
            for b in grouped.values()]
