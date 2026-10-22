"""기본 RAG 답변: 검색 → 근거 구성 → LLM(JSON) → 인용 검증.

    python -m rag.answer "서울 출장 숙박비 상한은?" [--k 8] [--show-context] [--json] [--model qwen3:8b]

거절은 두 단계로 판단한다.
1. 검색 top-1 점수가 MIN_SCORE 미만이면 LLM 을 부르지 않고 거절 (명백히 범위 밖인 질문)
2. 그 외에는 LLM 이 근거를 읽고 answerable 로 판단 — W2 기준선에서 거절 문항의 검색 점수(최대 0.75)가
   답변 가능 문항(최소 0.59)보다 높은 경우가 있어 점수만으로는 거절을 판단할 수 없음을 확인했다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field

from retrieval.refs import normalize_ref

from .definitions import definition_blocks, matching_definitions
from .llm import OllamaChat
from .numbers import ungrounded_numbers
from .verify import unsupported_periods, verify_quotes
from .prompts import ANSWER_SCHEMA, build_messages

MIN_SCORE = 0.45        # 이 점수 미만이면 검색 단계에서 바로 거절
MAX_BLOCKS = 5          # LLM 에 넣는 근거 조항 수
# 근거 총 글자 수. 실측상 한국어 약 0.85 token/글자 → 6,000자 ≈ 5.1K tokens.
# num_ctx 8K 에서 시스템 프롬프트(~0.6K) · 답변(~0.5K) 공간을 남긴다 (PRD 8.3)
CHAR_BUDGET = 6000
BLOCK_CAP = 1800        # 조항 하나가 예산을 독차지하지 않도록 조항당 상한
# 근거 오용 게이트: 인용한 조항의 리랭커 점수가 최고 후보의 이 비율 미만이면 "더 관련 있는 조항을 두고 엉뚱한 조항을 인용"으로 보고 거절.
# W3 측정: 답 있는 91문항의 정답 조항 비율 최소 0.32, 근거 오용 q010 0.088 · q027 0.137 (같은 Golden Set 으로 정함 — 과적합 주의)
RELEVANCE_RATIO = 0.2
REFUSAL_TEXT = "등록된 규정에서 이 질문에 대한 근거를 찾지 못했습니다. 담당 부서에 확인해 주세요."
UNVERIFIED_TEXT = "규정 원문에서 질문하신 내용을 확인하지 못해 답변하지 않습니다. 담당 부서에 확인해 주세요."


@dataclass
class AnswerResult:
    question: str
    answerable: bool
    answer: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    ungrounded_numbers: list[float] = field(default_factory=list)  # 질문 · 근거에 없는 숫자
    context: list[dict] = field(default_factory=list)   # [{"ref", "score", "chars"}]
    refused_by: str | None = None                        # "retrieval" | "llm" | "verification" | "error" | None
    quotes: list[dict] = field(default_factory=list)              # 원문 대조를 통과한 근거 인용문
    rejected_quotes: list[dict] = field(default_factory=list)     # 원문에 없는 인용문 (모델이 지어낸 문장)
    raw_answer: str = ""                                          # 검증 거절 전 모델 답 (디버깅 · 평가용)
    warnings: list[str] = field(default_factory=list)
    search_ms: float = 0.0
    llm_ms: float = 0.0
    prompt_tokens: int = 0


def keyword_overlap(question: str, text: str) -> int:
    """질문 어절이 본문에 나오는 정도. 조사가 붙은 어절("일반강사가")은 앞에서부터 줄여 가며(≥2자) 찾는다."""
    score = 0
    for token in re.findall(r"[가-힣A-Za-z0-9]+", question):
        for end in range(len(token), 1, -1):
            if token[:end] in text:
                score += end
                break
    return score


def _join_parts(parts: list[tuple[int, str]], cap: int, question: str | None = None) -> str:
    """상한 안에 드는 조각만 남겨 원래 순서로 잇는다.

    남길 조각의 우선순위: 질문 어절이 많이 나오는 조각 → 검색 순위.
    (W2 평가 q024: 검색 점수만 보면 "사외강사" 조각이 "사내 일반강사" 조각을 밀어냈다)
    """
    if question:
        ranked = sorted(enumerate(parts), key=lambda x: (-keyword_overlap(question, x[1][1]), x[0]))
        parts = [p for _, p in ranked]
    kept, used = [], 0
    for part_no, text in parts:
        if kept and used + len(text) > cap:
            continue
        kept.append((part_no, text[:cap] if not kept and len(text) > cap else text))
        used += len(kept[-1][1])
    kept.sort()
    out, prev = [], None
    for part_no, text in kept:
        if prev is not None and part_no != prev + 1:
            out.append("…(중략)")
        out.append(text)
        prev = part_no
    return "\n".join(out)


def select_blocks(hits: list, max_blocks: int = MAX_BLOCKS, char_budget: int = CHAR_BUDGET,
                  block_cap: int = BLOCK_CAP, question: str | None = None) -> list[dict]:
    """검색 결과를 조항 단위 근거 블록으로 묶는다.

    - 같은 조항의 분할 조각은 하나의 블록으로 합친다 (블록 순위는 처음 나온 조각 기준)
    - 조항당 block_cap 을 넘으면 검색 점수가 높은 조각부터 남긴다 (긴 조문 하나가 예산을 독차지하지 않게)
    - 전체가 char_budget 을 넘으면 뒤 블록을 자른다
    """
    order: list[str] = []
    grouped: dict[str, dict] = {}
    for hit in hits:
        c = hit.chunk
        key = normalize_ref(hit.ref)
        if key not in grouped:
            if len(order) >= max_blocks:
                continue
            order.append(key)
            grouped[key] = {"ref": hit.ref, "doc_title": c.get("doc_title", ""), "org": c.get("org", ""),
                            "effective_date": c.get("effective_date", ""), "score": hit.score, "parts": [],
                            "rerank": None}
        grouped[key]["parts"].append((c.get("part", 0), c["text"]))
        rr = getattr(hit, "rerank", None)
        if rr is not None:
            grouped[key]["rerank"] = max(rr, grouped[key]["rerank"] or 0.0)

    blocks, remaining = [], char_budget
    for key in order:
        g = grouped[key]
        text = _join_parts(g["parts"], block_cap, question)
        if remaining < 200:
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n…(이하 생략)"
        remaining -= len(text)
        blocks.append({"ref": g["ref"], "doc_title": g["doc_title"], "org": g["org"],
                       "effective_date": g["effective_date"], "score": g["score"], "rerank": g["rerank"],
                       "text": text})
    return blocks


def parse_answer(content: str) -> dict:
    data = json.loads(content)
    if not isinstance(data, dict) or not {"answerable", "answer", "citations"} <= data.keys():
        raise ValueError("필수 필드 누락")
    return data


_CONTEXT_NO_RE = re.compile(r"^\s*\[?\s*근거\s*(\d+)\s*\]?")
_ARTICLE_HEAD_RE = re.compile(r"^(.*?(?:제\s*\d+\s*조(?:\s*의\s*\d+)?|별\s*표\s*\d*(?:\s*의\s*\d+)?))")


def canonical_citation(cited: str, blocks: list[dict]) -> str | None:
    """모델이 쓴 인용 표기를 근거 블록의 참조로 맞춘다. 근거에 없으면 None.

    허용하는 변형 (조항은 같고 표기만 다른 경우):
    - 항 · 호 · 설명이 붙음: "회계규정 제15조 ②", "보수규정 제2조 제6항 (연봉월액 정의)" → "회계규정 제15조"
    - 근거 번호: "[근거 1] 별표 4 …" → 1번 근거의 참조
    - 규정명 생략: "제57조" → 근거 중 그 조항이 하나뿐이면 그 참조
    """
    available = {normalize_ref(b["ref"]): b["ref"] for b in blocks}
    if m := _CONTEXT_NO_RE.match(cited):
        n = int(m.group(1))
        return blocks[n - 1]["ref"] if 1 <= n <= len(blocks) else None
    text = re.sub(r"[(（][^)）]*[)）]", " ", cited).strip()
    head = _ARTICLE_HEAD_RE.match(text)
    key = normalize_ref(head.group(1) if head else text)
    if key in available:
        return available[key]
    if key.startswith(("제", "별표")):  # 규정명이 없는 "제57조" · "별표4" 형태
        matches = [ref for k, ref in available.items() if k.endswith(key)]
        if len(matches) == 1:
            return matches[0]
    return None


def validate_citations(cited: list[str], blocks: list[dict]) -> tuple[list[str], list[str]]:
    """인용이 실제로 제공한 근거 안에 있는지 확인 (없는 조항을 지어낸 인용을 걸러낸다)."""
    valid, invalid = [], []
    for c in cited:
        ref = canonical_citation(c, blocks)
        if ref is None:
            invalid.append(c)
        elif ref not in valid:
            valid.append(ref)
    return valid, invalid


def misused_evidence(quotes: list[dict], blocks: list[dict], ratio: float = RELEVANCE_RATIO) -> str | None:
    """인용한 조항이 질문과 관련이 낮으면(근거 오용) 사유 문자열, 아니면 None. 리랭커 점수가 없으면 판단하지 않는다.

    인용문 대조는 "그 문장이 원문에 있는가"만 본다. q010 처럼 주소 변경신고 기한을 묻는데 이전비 신청 기한(실제 원문)을
    인용하면 대조를 통과한다 → (질문, 인용 조항) 관련도를 최고 후보와 비교한다.
    """
    scored = {normalize_ref(b["ref"]): b["rerank"] for b in blocks if b.get("rerank") is not None}
    if not scored:
        return None
    top = max(scored.values())
    quoted = [scored[normalize_ref(q["ref"])] for q in quotes if normalize_ref(q["ref"]) in scored]
    if not quoted or top <= 0:
        return None  # 정의 블록만 인용했거나 점수 없음 → 판단 보류
    best = max(quoted)
    if best < ratio * top:
        return f"인용 조항의 질문 관련도가 낮음 (리랭커 {best:.3f} < 최고 후보 {top:.3f} × {ratio})"
    return None


def with_definitions(question: str, blocks: list[dict], definitions: dict | None, top_docs: int = 3) -> list[dict]:
    """상위 근거와 같은 규정의 정의 조항에서, 질문에 나온 용어의 정의 줄을 근거로 덧붙인다."""
    if not definitions or not blocks:
        return blocks
    docs = [b["doc_title"] for b in blocks[:top_docs] if b.get("doc_title")]
    present = {normalize_ref(b["ref"]) for b in blocks}
    defs = matching_definitions(question, docs, definitions, exclude_refs=present)
    return blocks + definition_blocks(defs)


def answer_question(question: str, search, llm, k: int = 8, min_score: float = MIN_SCORE,
                    definitions: dict | None = None) -> AnswerResult:
    """search(question, k) → [Hit], llm.chat(messages, schema=...) → LLMResponse

    definitions: rag.definitions.build_index(chunks) 결과. 주면 질문 속 정의 용어의 정의 줄을 근거에 덧붙인다.
    """
    started = time.perf_counter()
    hits = search(question, k)
    search_ms = (time.perf_counter() - started) * 1000

    if not hits or hits[0].score < min_score:
        return AnswerResult(question, False, REFUSAL_TEXT, refused_by="retrieval", search_ms=search_ms,
                            context=[{"ref": h.ref, "score": round(h.score, 4)} for h in hits[:3]])

    blocks = with_definitions(question, select_blocks(hits, question=question), definitions)
    messages = build_messages(question, blocks)
    data, resp, warnings = None, None, []
    for attempt in range(2):  # JSON 파싱 실패 · 호출 실패 시 1회 재시도 (PRD F-AGT-04)
        try:
            resp = llm.chat(messages, schema=ANSWER_SCHEMA, temperature=0.1 if attempt == 0 else 0.0)
        except RuntimeError as exc:
            warnings.append(f"LLM 호출 실패 (시도 {attempt + 1}): {exc}")
            continue
        try:
            data = parse_answer(resp.content)
            break
        except (ValueError, json.JSONDecodeError):
            # 출력 상한(num_predict)에 걸려 JSON 이 잘린 경우도 여기로 온다
            warnings.append(f"LLM 응답 JSON 파싱 실패 (시도 {attempt + 1}, 출력 {resp.output_tokens} tokens)")
    context = [{"ref": b["ref"], "score": round(b["score"], 4), "chars": len(b["text"]),
                "kind": b.get("kind", "search")} for b in blocks]
    if data is None:
        return AnswerResult(question, False, "답변 생성에 실패했습니다. 잠시 후 다시 시도해 주세요.",
                            context=context, refused_by="error", warnings=warnings, search_ms=search_ms,
                            llm_ms=resp.total_ms if resp else 0.0)

    citations, invalid = validate_citations([str(c) for c in data.get("citations", [])], blocks)
    answerable = bool(data["answerable"])
    answer_text = str(data["answer"]).strip()
    quotes, rejected = verify_quotes([q for q in data.get("quotes", []) if isinstance(q, dict)], blocks)
    if rejected:
        warnings.append(f"원문에 없는 인용문 {len(rejected)}개")
    refused_by = None if answerable else "llm"
    raw_answer = answer_text
    if answerable:  # 모델이 답하겠다고 해도, 근거 문장이 원문에 있고 기한 · 기간이 그 문장에 있어야 답한다
        missing_periods = unsupported_periods(answer_text, question, quotes)
        misused = misused_evidence(quotes, blocks)
        if not quotes:
            reason = "검증된 근거 인용문 없음"
        elif missing_periods:
            reason = f"근거 인용문에 없는 기한 · 기간 {missing_periods}"
        elif misused:
            reason = misused
        else:
            reason = None
        if reason:
            warnings.append(f"검증 거절: {reason}")
            answerable, refused_by, answer_text = False, "verification", UNVERIFIED_TEXT
    if invalid:
        warnings.append(f"근거에 없는 조항을 인용함: {invalid}")
    if answerable and not citations:
        warnings.append("답변했지만 유효한 인용이 없음")
    unknown = ungrounded_numbers(answer_text, question, [b["text"] for b in blocks]) if answerable else []
    if unknown:  # 지어낸 숫자 또는 계산 결과 — 계산은 W4 계산 도구가 맡는다
        warnings.append(f"근거에 없는 숫자: {[f'{n:,.2f}'.rstrip('0').rstrip('.') for n in unknown]} (계산 결과가 아니면 환각)")
    return AnswerResult(
        question=question, answerable=answerable, answer=answer_text,
        citations=citations if answerable else [], invalid_citations=invalid, ungrounded_numbers=unknown,
        context=context, quotes=quotes, rejected_quotes=rejected, raw_answer=raw_answer,
        refused_by=refused_by, warnings=warnings,
        search_ms=search_ms, llm_ms=resp.total_ms, prompt_tokens=resp.prompt_tokens,
    )


def render(result: AnswerResult, blocks_meta: dict[str, dict] | None = None) -> str:
    lines = [f"Q: {result.question}", "", f"A: {result.answer}"]
    if result.citations:
        lines += ["", "출처: " + " · ".join(result.citations)]
    if not result.answerable:
        reason = {"retrieval": "검색 점수 낮음", "llm": "근거 부족 — LLM 판단",
                  "verification": "근거 인용 검증 실패", "error": "생성 오류"}.get(result.refused_by, "")
        lines += ["", f"(거절: {reason})"]
    for q in result.quotes:
        lines.append(f"  📎 [{q['ref']}] {q['text']}")
    for w in result.warnings:
        lines.append(f"⚠️ {w}")
    lines += ["", "근거 후보: " + " | ".join(
                  f"{c['ref']} " + ("(정의)" if c.get("kind") == "definition" else f"{c['score']:.3f}")
                  for c in result.context),
              f"(검색 {result.search_ms:.0f}ms · 생성 {result.llm_ms / 1000:.1f}s · 입력 {result.prompt_tokens} tokens)"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="규정 근거 기반 답변")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=8, help="검색 후보 수 (조각 포함)")
    parser.add_argument("--model", help="Ollama 모델 (기본: .env 의 LLM_MODEL)")
    parser.add_argument("--show-context", action="store_true", help="LLM 에 넣은 근거 원문 출력")
    parser.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    args = parser.parse_args(argv)

    from retrieval.index import load_chunks
    from retrieval.search import Searcher

    from .definitions import build_index

    searcher = Searcher()
    searcher.warmup()
    definitions = build_index(load_chunks(searcher.settings.chunks_path))
    llm = OllamaChat(model=args.model)
    try:
        hits_cache = {}

        def search(q, k):
            hits_cache["hits"] = searcher.search(q, k)
            return hits_cache["hits"]

        result = answer_question(args.question, search, llm, k=args.k, definitions=definitions)
    finally:
        searcher.close()

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=1))
        return 0
    print(render(result))
    if args.show_context and hits_cache.get("hits"):
        print("\n===== 근거 원문")
        for b in with_definitions(args.question, select_blocks(hits_cache["hits"], question=args.question), definitions):
            print(f"\n[{b['ref']}] ({b['score']:.3f}{', 정의 첨부' if b.get('kind') == 'definition' else ''})\n{b['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
