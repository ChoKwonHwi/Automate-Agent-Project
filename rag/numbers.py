"""한국어 텍스트에서 숫자 추출 (금액 · 일수 · 비율). 답변 검증과 평가에서 함께 쓴다."""
from __future__ import annotations

import re

# 조항 번호 · 별표 번호 · 날짜 · 원문자 항 번호는 금액/일수가 아니므로 먼저 지운다
_NOISE_RE = re.compile(
    r"제\s*\d+\s*(?:조|항|호|장|절|목)(?:\s*의\s*\d+)?"
    r"|별\s*표\s*\d+(?:\s*의\s*\d+)?"
    r"|\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?"
)
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(억|천만|백만|만|천)?")
_UNITS = {"억": 1e8, "천만": 1e7, "백만": 1e6, "만": 1e4, "천": 1e3}


def extract_numbers(text: str) -> list[float]:
    """'25,000원', '10만원', '3천만원', '166,666.67' → 숫자."""
    text = _NOISE_RE.sub(" ", text or "")
    out = []
    for num, unit in _NUMBER_RE.findall(text):
        try:
            value = float(num.replace(",", ""))
        except ValueError:
            continue
        out.append(value * _UNITS.get(unit, 1))
    return out


def contains_number(values: list[float], target: float, tol: float = 0.5) -> bool:
    return any(abs(v - target) < tol for v in values)


def ungrounded_numbers(answer: str, question: str, context_texts: list[str]) -> list[float]:
    """답변의 숫자 중 질문에도 근거에도 없는 것 (지어낸 숫자 또는 계산 결과)."""
    known = extract_numbers(question)
    for t in context_texts:
        known += extract_numbers(t)
    out = []
    for n in extract_numbers(answer):
        if not contains_number(known, n) and not contains_number(out, n):
            out.append(n)
    return out
