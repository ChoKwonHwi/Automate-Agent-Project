"""답변 검증: 모델이 제시한 근거 인용문(quotes)을 근거 원문과 코드로 대조한다.

W2 평가에서 프롬프트로 "추측하지 마라"고 해도 모델이 기한을 지어냈다 (q003 "30일 이내" → "8근무시간 이내").
모델의 판단 대신, 답의 근거 문장이 원문에 실제로 있는지와 답의 기한 · 기간이 그 문장에 있는지를 코드로 확인한다.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .numbers import extract_numbers

MIN_QUOTE_CHARS = 8      # 너무 짧은 인용("제5조")은 아무 데나 맞으므로 인정하지 않는다
FUZZY_RATIO = 0.85       # 공백 · 기호 차이 외의 사소한 차이(조사 하나 등)는 허용

_SAME_CHARS = str.maketrans({"ㆍ": "·", "・": "·", "∙": "·", "•": "·", "“": '"', "”": '"', "‘": "'", "’": "'",
                             "「": '"', "」": '"', "『": '"', "』": '"', "（": "(", "）": ")"})
# 기한 · 기간을 나타내는 숫자 (금액 합계 같은 계산값은 W4 계산 도구가 맡으므로 여기서 보지 않는다)
_PERIOD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(근무시간|시간|개월|주일|주|일|년)")


def normalize(text: str) -> str:
    text = (text or "").translate(_SAME_CHARS)
    text = text.replace("…(중략)", "").replace("(계속)", "")
    return re.sub(r"[\s|]+", "", text)


def quote_in_text(quote: str, text: str) -> bool:
    q, t = normalize(quote), normalize(text)
    if len(q) < MIN_QUOTE_CHARS:
        return False
    if q in t:
        return True
    # 가장 긴 일치 구간을 기준점으로, 원문에서 인용문 길이만큼의 창을 잘라 그 창과 인용문 전체를 비교한다
    # (원문 전체와 비교하면 흩어진 글자들이 우연히 맞아 지어낸 문장도 통과할 수 있다)
    anchor = SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
    if anchor.size == 0:
        return False
    start = max(0, anchor.a - anchor.b - 5)
    window = t[start:anchor.a - anchor.b + len(q) + 5]
    matched = sum(m.size for m in SequenceMatcher(None, window, q, autojunk=False).get_matching_blocks())
    return matched / len(q) >= FUZZY_RATIO


def verify_quotes(quotes: list[dict], blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """(원문에 있는 인용, 없는 인용). ref 가 틀려도 다른 근거 블록에 있으면 인정하고 ref 를 바로잡는다."""
    verified, rejected = [], []
    for q in quotes:
        text = str(q.get("text", ""))
        ref = str(q.get("ref", ""))
        same = [b for b in blocks if normalize(b["ref"]) == normalize(ref)]
        others = [b for b in blocks if b not in same]
        found = next((b for b in same + others if quote_in_text(text, b["text"])), None)
        if found:
            verified.append({"ref": found["ref"], "text": text})
        else:
            rejected.append({"ref": ref, "text": text})
    return verified, rejected


def period_mentions(text: str) -> list[tuple[float, str]]:
    return [(float(n), unit) for n, unit in _PERIOD_RE.findall(text or "")]


def unsupported_periods(answer: str, question: str, quotes: list[dict]) -> list[str]:
    """답에 나온 기한 · 기간 중 검증된 인용문에도, 질문에도 없는 것. 예: "8근무시간"."""
    asked = {n for n, _ in period_mentions(question)}
    quoted = extract_numbers(" ".join(q["text"] for q in quotes))
    missing = []
    for n, unit in period_mentions(answer):
        if n in asked:
            continue
        if not any(abs(n - v) < 0.5 for v in quoted):
            label = f"{n:g}{unit}"
            if label not in missing:
                missing.append(label)
    return missing
