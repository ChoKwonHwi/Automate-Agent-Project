"""페이지 텍스트 정제: 쪽번호 · 반복 머리말/꼬리말 제거, 공백 정규화.

결과는 (줄, 페이지번호) 목록으로 반환해 이후 청크의 원문 페이지를 추적할 수 있게 한다.
"""
from __future__ import annotations

import re
from collections import Counter

from .loaders import Page

Line = tuple[str, int]

_PAGE_NUMBER_RE = re.compile(
    r"^(?:[-–—]?\s*\d{1,4}\s*[-–—]?|\d{1,4}\s*/\s*\d{1,4}|\(\s*\d{1,4}\s*\)|page\s*\d{1,4})$",
    re.IGNORECASE,
)
# 국가법령정보센터(law.go.kr)에서 저장한 PDF의 쪽 꼬리말: "법제처 3 국가법령정보센터"
_LAW_GO_KR_FOOTER_RE = re.compile(r"^법제처\s*\d+\s*국가법령정보센터$")
_SPACES_RE = re.compile(r"[ \t 　]+")
# 짧은 문서에서 조 · 부칙 · 별표 헤딩이 머리말로 오인되지 않게 한다 (숫자를 #로 바꿔 비교하므로)
_ARTICLE_RE = re.compile(r"^(?:제\s*\d+\s*조|부\s*칙|[\[【<〔]?\s*별\s*[표지])")


def normalize_line(line: str) -> str:
    line = line.replace("﻿", "")
    if line.lstrip().startswith("|"):
        return line.strip()
    return _SPACES_RE.sub(" ", line).strip()


def _header_key(line: str) -> str:
    return re.sub(r"\d+", "#", line)


def clean_pages(pages: list[Page], header_ratio: float = 0.6) -> list[Line]:
    page_lines = [[normalize_line(l) for l in p.text.splitlines()] for p in pages]
    page_lines = [[l for l in lines if l] for lines in page_lines]

    repeated: set[str] = set()
    if len(pages) >= 2:
        counter: Counter[str] = Counter()
        for lines in page_lines:
            edges = {_header_key(l) for l in lines[:2] + lines[-2:] if not l.startswith("|")}
            counter.update(edges)
        repeated = {k for k, c in counter.items() if c >= len(pages) * header_ratio}

    result: list[Line] = []
    for page, lines in zip(pages, page_lines):
        last = len(lines) - 1
        for idx, line in enumerate(lines):
            if _PAGE_NUMBER_RE.match(line) or _LAW_GO_KR_FOOTER_RE.match(line):
                continue
            is_edge = idx < 2 or idx > last - 2
            if is_edge and _header_key(line) in repeated and not _ARTICLE_RE.match(line):
                continue
            result.append((line, page.number))
    return result
