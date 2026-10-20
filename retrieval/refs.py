"""조항 참조 문자열 ("여비규정 제16조", "여비규정 별표1")과 청크 사이의 변환.

Golden Set 의 근거 · 검색 평가 · 답변 인용이 모두 이 표기를 쓴다.
chunk_id 대신 "규정명 + 조번호"를 쓰므로 재청킹해도 참조가 깨지지 않는다.
"""
from __future__ import annotations

import re

_ANNEX_SUFFIX_RE = re.compile(r"\s*별\s*표\s*[\d의]*$")


def chunk_ref(chunk: dict) -> str:
    """청크의 표준 참조. 분할 조각(part)은 모두 같은 참조를 가진다."""
    title = chunk["doc_title"].strip()
    label = chunk["article_label"].strip()
    if chunk["section"] == "별표":
        base = _ANNEX_SUFFIX_RE.sub("", title) or title
        return f"{base} {re.sub(r'\s+', '', label)}"
    if chunk["section"] == "전문":
        return f"{title} 전문"
    return f"{title} {label}"


def normalize_ref(ref: str) -> str:
    return re.sub(r"\s+", "", ref)


def split_refs(text: str) -> list[str]:
    """`"여비규정 별표1; 여비규정 제16조"` → 참조 목록."""
    return [r.strip() for r in re.split(r"[;\n]", text or "") if r.strip()]
