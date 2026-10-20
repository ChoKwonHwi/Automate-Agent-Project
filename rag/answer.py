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
import sys
import time
from dataclasses import asdict, dataclass, field

from retrieval.refs import normalize_ref

from .llm import OllamaChat
from .prompts import ANSWER_SCHEMA, build_messages

MIN_SCORE = 0.45        # 이 점수 미만이면 검색 단계에서 바로 거절
MAX_BLOCKS = 5          # LLM 에 넣는 근거 조항 수
CHAR_BUDGET = 4500      # 근거 총 글자 수 (num_ctx 8K 안에서 시스템 프롬프트 · 답변 공간 확보, PRD 8.3)
REFUSAL_TEXT = "등록된 규정에서 이 질문에 대한 근거를 찾지 못했습니다. 담당 부서에 확인해 주세요."


@dataclass
class AnswerResult:
    question: str
    answerable: bool
    answer: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    context: list[dict] = field(default_factory=list)   # [{"ref", "score", "chars"}]
    refused_by: str | None = None                        # "retrieval" | "llm" | None
    warnings: list[str] = field(default_factory=list)
    search_ms: float = 0.0
    llm_ms: float = 0.0
    prompt_tokens: int = 0


def select_blocks(hits: list, max_blocks: int = MAX_BLOCKS, char_budget: int = CHAR_BUDGET) -> list[dict]:
    """검색 결과를 조항 단위 근거 블록으로 묶는다.

    같은 조항의 분할 조각은 하나의 블록으로 합치고(순위는 처음 나온 조각 기준), 글자 예산을 넘으면 뒤를 자른다.
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
            grouped[key] = {"ref": hit.ref, "org": c.get("org", ""), "effective_date": c.get("effective_date", ""),
                            "score": hit.score, "parts": {}}
        grouped[key]["parts"][c.get("part", 0)] = c["text"]

    blocks, remaining = [], char_budget
    for key in order:
        g = grouped[key]
        text = "\n".join(g["parts"][p] for p in sorted(g["parts"]))
        if remaining < 200:
            break
        if len(text) > remaining:
            text = text[:remaining] + "\n…(이하 생략)"
        remaining -= len(text)
        blocks.append({"ref": g["ref"], "org": g["org"], "effective_date": g["effective_date"],
                       "score": g["score"], "text": text})
    return blocks


def parse_answer(content: str) -> dict:
    data = json.loads(content)
    if not isinstance(data, dict) or not {"answerable", "answer", "citations"} <= data.keys():
        raise ValueError("필수 필드 누락")
    return data


def validate_citations(cited: list[str], blocks: list[dict]) -> tuple[list[str], list[str]]:
    """인용이 실제로 제공한 근거 안에 있는지 확인 (없는 조항을 지어낸 인용을 걸러낸다)."""
    available = {normalize_ref(b["ref"]): b["ref"] for b in blocks}
    valid, invalid = [], []
    for c in cited:
        key = normalize_ref(c)
        if key in available:
            if available[key] not in valid:
                valid.append(available[key])
        else:
            invalid.append(c)
    return valid, invalid


def answer_question(question: str, search, llm, k: int = 8, min_score: float = MIN_SCORE) -> AnswerResult:
    """search(question, k) → [Hit], llm.chat(messages, schema=...) → LLMResponse"""
    started = time.perf_counter()
    hits = search(question, k)
    search_ms = (time.perf_counter() - started) * 1000

    if not hits or hits[0].score < min_score:
        return AnswerResult(question, False, REFUSAL_TEXT, refused_by="retrieval", search_ms=search_ms,
                            context=[{"ref": h.ref, "score": round(h.score, 4)} for h in hits[:3]])

    blocks = select_blocks(hits)
    messages = build_messages(question, blocks)
    data, resp, warnings = None, None, []
    for attempt in range(2):  # JSON 파싱 실패 시 1회 재시도 (PRD F-AGT-04)
        resp = llm.chat(messages, schema=ANSWER_SCHEMA, temperature=0.1 if attempt == 0 else 0.0)
        try:
            data = parse_answer(resp.content)
            break
        except (ValueError, json.JSONDecodeError):
            warnings.append(f"LLM 응답 JSON 파싱 실패 (시도 {attempt + 1})")
    context = [{"ref": b["ref"], "score": round(b["score"], 4), "chars": len(b["text"])} for b in blocks]
    if data is None:
        return AnswerResult(question, False, "답변 생성에 실패했습니다. 잠시 후 다시 시도해 주세요.",
                            context=context, refused_by="error", warnings=warnings, search_ms=search_ms,
                            llm_ms=resp.total_ms if resp else 0.0)

    citations, invalid = validate_citations([str(c) for c in data.get("citations", [])], blocks)
    answerable = bool(data["answerable"])
    if invalid:
        warnings.append(f"근거에 없는 조항을 인용함: {invalid}")
    if answerable and not citations:
        warnings.append("답변했지만 유효한 인용이 없음")
    return AnswerResult(
        question=question, answerable=answerable, answer=str(data["answer"]).strip(),
        citations=citations if answerable else [], invalid_citations=invalid, context=context,
        refused_by=None if answerable else "llm", warnings=warnings,
        search_ms=search_ms, llm_ms=resp.total_ms, prompt_tokens=resp.prompt_tokens,
    )


def render(result: AnswerResult, blocks_meta: dict[str, dict] | None = None) -> str:
    lines = [f"Q: {result.question}", "", f"A: {result.answer}"]
    if result.citations:
        lines += ["", "출처: " + " · ".join(result.citations)]
    if not result.answerable:
        lines += ["", f"(거절: {'검색 점수 낮음' if result.refused_by == 'retrieval' else '근거 부족 — LLM 판단'})"]
    for w in result.warnings:
        lines.append(f"⚠️ {w}")
    lines += ["", "근거 후보: " + " | ".join(f"{c['ref']} {c['score']:.3f}" for c in result.context),
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

    from retrieval.search import Searcher

    searcher = Searcher()
    llm = OllamaChat(model=args.model)
    try:
        hits_cache = {}

        def search(q, k):
            hits_cache["hits"] = searcher.search(q, k)
            return hits_cache["hits"]

        result = answer_question(args.question, search, llm, k=args.k)
    finally:
        searcher.close()

    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=1))
        return 0
    print(render(result))
    if args.show_context and hits_cache.get("hits"):
        print("\n===== 근거 원문")
        for b in select_blocks(hits_cache["hits"]):
            print(f"\n[{b['ref']}] ({b['score']:.3f})\n{b['text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
