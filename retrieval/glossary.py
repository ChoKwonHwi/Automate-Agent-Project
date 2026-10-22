"""일상어 → 규정 용어 사전으로 검색어를 넓힌다 (LLM 없음, 지연 ~0ms).

W3 실험에서 LLM(Qwen3 8B) 쿼리 재작성은 "그만둔 다음에 회사 일 얘기해도 돼?"를 "퇴직 후 업무 참여"로만 해석해
정답(비밀 누설 금지)을 찾지 못했고, 호출마다 3~4초가 들었다. 사내 검색에서 흔히 쓰는 방식대로
사람이 관리하는 동의어 사전(data/glossary.csv)으로 대신한다.

- 사전은 특정 평가 질문이 아니라 일반적인 직장 구어 표현으로 작성했다.
- 효과 검증은 사람이 쓴 '일상어' 문항(Golden Set notes 에 "일상어")으로 따로 한다.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import ROOT

GLOSSARY_PATH = ROOT / "data/glossary.csv"


@dataclass(frozen=True)
class Entry:
    patterns: tuple[str, ...]
    expansion: str


def _key(text: str) -> str:
    return re.sub(r"\s+", "", text)


@lru_cache(maxsize=4)
def load_glossary(path: Path = GLOSSARY_PATH) -> tuple[Entry, ...]:
    if not path.exists():
        return ()
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return tuple(Entry(tuple(_key(p) for p in r["colloquial"].split("|") if p.strip()), r["expansion"].strip())
                 for r in rows if r.get("colloquial") and r.get("expansion"))


def expand(query: str, entries: tuple[Entry, ...] | None = None) -> list[str]:
    """질문에 나온 구어 표현에 해당하는 규정 용어 묶음들. 없으면 []."""
    q = _key(query)
    entries = load_glossary() if entries is None else entries
    return [e.expansion for e in entries if any(p and p in q for p in e.patterns)]
