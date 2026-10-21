"""답변 평가 + LLM 비교.

    python -m eval.answer                                   # .env 의 LLM_MODEL 하나
    python -m eval.answer --models qwen3:8b exaone3.5:7.8b gemma3:4b --name w2_llm_compare

자동 채점 (문항 유형별):
- 거절 정확도   : refuse 문항을 거절했는가
- 오거절률      : 답할 수 있는 문항을 거절했는가
- 인용 적중     : 인용 중 하나 이상이 정답 근거 조항인가 / 정밀도 · 재현율
- 수치 정답     : numeric 문항에서 정답의 핵심 숫자(질문에 이미 나온 숫자 제외)가 답에 모두 있는가 (오차 < 0.5)
- 지연 · 토큰   : LLM 생성 시간 (모델 로드 제외)
문장형 정답(fact · multi)의 내용 정확성은 자동 판정하지 않는다 → 사람 판정 시트(_review.md) 로 확인.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime

from retrieval.config import ROOT
from retrieval.refs import normalize_ref

from rag.numbers import extract_numbers

from .golden import GoldenItem, load_golden

REPORT_DIR = ROOT / "eval/reports"

def key_numbers(item: GoldenItem) -> list[float]:
    """정답의 숫자 중 질문에 이미 나온 숫자를 뺀 것 (질문을 따라 말한 숫자는 채점하지 않음)."""
    asked = extract_numbers(item.question)
    keys = []
    for n in extract_numbers(item.answer):
        if not any(abs(n - a) < 0.5 for a in asked) and not any(abs(n - k) < 0.5 for k in keys):
            keys.append(n)
    return keys


def numeric_score(item: GoldenItem, answer: str) -> tuple[float | None, list[float]]:
    """(맞힌 핵심 숫자 비율, 빠진 숫자). 채점할 숫자가 없으면 (None, [])."""
    keys = key_numbers(item)
    if not keys:
        return None, []
    got = extract_numbers(answer)
    missing = [k for k in keys if not any(abs(k - g) < 0.5 for g in got)]
    return 1 - len(missing) / len(keys), missing


def conclusion_ok(item: GoldenItem, answer: str) -> bool | None:
    """정답 문장의 마지막 핵심 숫자(보통 결론값)가 답에 있는가. 조건으로 언급된 숫자는 따지지 않는 느슨한 기준."""
    keys = key_numbers(item)
    if not keys:
        return None
    return any(abs(keys[-1] - g) < 0.5 for g in extract_numbers(answer))


def citation_scores(cited: list[str], evidence: list[str]) -> dict:
    c = {normalize_ref(x) for x in cited}
    e = {normalize_ref(x) for x in evidence}
    inter = c & e
    return {"hit": bool(inter), "precision": len(inter) / len(c) if c else 0.0,
            "recall": len(inter) / len(e) if e else 0.0}


def grade(item: GoldenItem, result) -> dict:
    row = {"id": item.id, "type": item.type, "category": item.category, "answerable": result.answerable,
           "refused_by": result.refused_by, "answer": result.answer, "citations": result.citations,
           "invalid_citations": result.invalid_citations, "warnings": result.warnings,
           "ungrounded_numbers": result.ungrounded_numbers,
           "raw_answer": getattr(result, "raw_answer", ""), "quotes": getattr(result, "quotes", []),
           "rejected_quotes": getattr(result, "rejected_quotes", []),
           "llm_ms": round(result.llm_ms), "prompt_tokens": result.prompt_tokens,
           "context": [c["ref"] for c in result.context]}
    if item.type == "refuse":
        row["correct_refusal"] = not result.answerable
        return row
    row["false_refusal"] = not result.answerable
    if result.answerable:
        row.update(citation_scores(result.citations, item.evidence))
    if item.type == "numeric":
        row["numeric"], row["numeric_missing"] = numeric_score(item, result.answer)
        row["conclusion"] = conclusion_ok(item, result.answer)
    return row


def summarize(rows: list[dict]) -> dict:
    refuse = [r for r in rows if r["type"] == "refuse"]
    ans = [r for r in rows if r["type"] != "refuse"]
    answered = [r for r in ans if r["answerable"]]
    numeric = [r for r in ans if r["type"] == "numeric" and r.get("numeric") is not None]
    llm = sorted(r["llm_ms"] for r in rows if r["llm_ms"])

    def ratio(xs, pred):
        return sum(1 for x in xs if pred(x)) / len(xs) if xs else None

    return {
        "n": len(rows),
        "refusal_accuracy": ratio(refuse, lambda r: r["correct_refusal"]),
        "false_refusal_rate": ratio(ans, lambda r: r["false_refusal"]),
        "citation_hit": ratio(answered, lambda r: r["hit"]),
        "citation_precision": statistics.mean(r["precision"] for r in answered) if answered else None,
        "citation_recall": statistics.mean(r["recall"] for r in answered) if answered else None,
        "numeric_exact": ratio(numeric, lambda r: r["numeric"] == 1.0),
        "numeric_conclusion": ratio(numeric, lambda r: bool(r.get("conclusion"))),
        "ungrounded_items": sum(1 for r in rows if r.get("ungrounded_numbers")),
        "invalid_citation_items": sum(1 for r in rows if r["invalid_citations"]),
        "errors": sum(1 for r in rows if r["refused_by"] == "error"),
        "verification_refusals": sum(1 for r in rows if r["refused_by"] == "verification"),
        "llm_p50_ms": statistics.median(llm) if llm else None,
        "llm_p95_ms": llm[max(0, int(len(llm) * 0.95) - 1)] if llm else None,
        "avg_prompt_tokens": statistics.mean(r["prompt_tokens"] for r in rows if r["prompt_tokens"]) if llm else None,
    }


def _fmt(v, pct=True) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.0f}%" if pct else f"{v:,.0f}"


def build_report(name: str, items: list[GoldenItem], runs: dict[str, list[dict]], meta: dict) -> str:
    sums = {m: summarize(rows) for m, rows in runs.items()}
    models = list(runs)
    out = [f"# 답변 평가 — {name}", "", f"- 생성: {datetime.now():%Y-%m-%d %H:%M}",
           f"- 설정: {meta['note']}", f"- 문항: {len(items)}개 "
           f"(refuse {sum(1 for i in items if i.type == 'refuse')}, numeric {sum(1 for i in items if i.type == 'numeric')})",
           "", "## 모델 비교", "",
           "| 지표 | 목표(PRD) | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    rows = [
        ("거절 정확도 (refuse 문항 거절)", "≥ 90%", "refusal_accuracy", True),
        ("오거절률 (답할 수 있는데 거절) ↓", "-", "false_refusal_rate", True),
        ("인용 적중 (정답 조항 1개 이상 인용)", "≥ 90%", "citation_hit", True),
        ("인용 정밀도", "-", "citation_precision", True),
        ("인용 재현율 (정답 조항 중 인용 비율)", "-", "citation_recall", True),
        ("수치 정답 · 엄격 (numeric, 정답의 숫자 모두 일치)", "100% (W4 계산도구)", "numeric_exact", True),
        ("수치 정답 · 결론 (numeric, 결론 숫자 일치)", "-", "numeric_conclusion", True),
        ("근거에 없는 숫자를 쓴 답변 (문항 수) ↓", "0 (계산값 제외)", "ungrounded_items", False),
        ("근거에 없는 조항 인용 (문항 수) ↓", "0", "invalid_citation_items", False),
        ("응답 오류 (JSON 실패) ↓", "0", "errors", False),
        ("검증 단계 거절 (문항 수, 인용문 대조 실패)", "-", "verification_refusals", False),
        ("생성 지연 p50 (ms) ↓", "-", "llm_p50_ms", False),
        ("생성 지연 p95 (ms) ↓", "≤ 10,000", "llm_p95_ms", False),
        ("평균 입력 토큰", "-", "avg_prompt_tokens", False),
    ]
    for label, goal, key, pct in rows:
        out.append(f"| {label} | {goal} | " + " | ".join(_fmt(sums[m][key], pct) for m in models) + " |")

    out += ["", "## 문항별", "", "기호: ✅ 적중/정답 · 🟡 결론 숫자만 일치 · ❌ 틀림 · 🚫 거절 · ⚠️ 근거 밖 인용 · 🔢 근거에 없는 숫자 · `-` 채점 안 함", "",
            "| id | type | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    by_id = {m: {r["id"]: r for r in rows_} for m, rows_ in runs.items()}
    for item in items:
        cells = []
        for m in models:
            r = by_id[m][item.id]
            if item.type == "refuse":
                cell = "✅ 거절" if r["correct_refusal"] else "❌ 답변함"
            elif not r["answerable"]:
                cell = "🚫 오거절"
            else:
                cell = ("✅" if r["hit"] else "❌") + " 인용"
                if item.type == "numeric":
                    if r.get("numeric") is None:
                        cell += " · -"
                    else:
                        cell += " · " + ("✅" if r["numeric"] == 1.0 else ("🟡" if r.get("conclusion") else "❌")) + " 수치"
            if r["invalid_citations"]:
                cell += " ⚠️"
            if r.get("ungrounded_numbers"):
                cell += " 🔢"
            cells.append(cell)
        out.append(f"| {item.id} | {item.type} | " + " | ".join(cells) + " |")

    problems = []
    for m in models:
        for r in runs[m]:
            if r["type"] == "refuse" and not r["correct_refusal"]:
                problems.append(f"- **{m}** `{r['id']}` 거절해야 하는데 답변: {r['answer'][:120]}")
            elif r["type"] != "refuse" and r["false_refusal"]:
                how = "검증 거절" if r["refused_by"] == "verification" else "오거절"
                shown = r.get("raw_answer") or r["answer"]
                problems.append(f"- **{m}** `{r['id']}` {how}: {shown[:120]} (근거: {', '.join(r['context'][:3])})")
            elif r.get("ungrounded_numbers"):
                problems.append(f"- **{m}** `{r['id']}` 근거에 없는 숫자 {r['ungrounded_numbers']}: {r['answer'][:120]}")
            elif r.get("numeric") is not None and r["numeric"] < 1.0:
                problems.append(f"- **{m}** `{r['id']}` 수치 누락 {r['numeric_missing']}: {r['answer'][:120]}")
            elif r["type"] != "refuse" and r["answerable"] and not r["hit"]:
                problems.append(f"- **{m}** `{r['id']}` 인용 불일치: 인용 {r['citations']} / 근거 {', '.join(r['context'][:3])}")
    out += ["", f"## 확인할 문항 ({len(problems)})", ""] + (problems or ["(없음)"])
    out += ["", "문장형 정답의 내용 정확성은 자동 채점하지 않습니다 → "
                f"[`{name}_review.md`]({name}_review.md) 에서 사람이 판정합니다."]
    return "\n".join(out) + "\n"


def build_review(name: str, items: list[GoldenItem], runs: dict[str, list[dict]]) -> str:
    models = list(runs)
    by_id = {m: {r["id"]: r for r in rows} for m, rows in runs.items()}
    out = [f"# 답변 사람 판정 — {name}", "",
           "각 모델 답을 정답과 비교해 `[ ]` 를 `[o]`(맞음) · `[~]`(일부 맞음) · `[x]`(틀림) 로 바꾼다.",
           "판단 기준: 정답의 핵심 내용(결론 · 숫자 · 조건)이 있는가, 정답에 없는 틀린 내용을 덧붙이지 않았는가.", ""]
    for item in items:
        out += [f"## {item.id} · {item.type} · {item.category}", "", f"**Q.** {item.question}", "",
                f"**정답** ({'; '.join(item.evidence) or '거절'}): {item.answer}", ""]
        for m in models:
            r = by_id[m][item.id]
            cite = f" — 인용: {', '.join(r['citations'])}" if r["citations"] else ""
            tag = "" if r["answerable"] else (" 🚫검증거절" if r["refused_by"] == "verification" else " 🚫거절")
            shown = r["answer"]
            if r["refused_by"] == "verification" and r.get("raw_answer"):
                shown += f" — (거절 전 모델 답: {r['raw_answer']})"
            out.append(f"- [ ] **{m}**{tag}: {shown}{cite}")
        out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Golden Set 답변 평가 · LLM 비교")
    parser.add_argument("--models", nargs="+", help="Ollama 모델들 (기본: .env 의 LLM_MODEL)")
    parser.add_argument("--name", default="w2_llm_compare")
    parser.add_argument("--ids", nargs="*", help="일부 문항만 (예: q001 q005)")
    args = parser.parse_args(argv)

    from rag.answer import answer_question
    from rag.definitions import build_index
    from rag.llm import OllamaChat
    from retrieval.index import load_chunks
    from retrieval.search import Searcher

    items = load_golden()
    if args.ids:
        items = [i for i in items if i.id in set(args.ids)]
    models = args.models or [OllamaChat().model]

    searcher = Searcher()
    searcher.warmup()
    definitions = build_index(load_chunks(searcher.settings.chunks_path))
    runs: dict[str, list[dict]] = {}
    try:
        for m in models:
            for other in models:  # 다른 모델은 GPU 에서 내린다 (VRAM 8GB)
                if other != m:
                    OllamaChat(model=other).unload()
            llm = OllamaChat(model=m)
            load_ms = llm.warmup()
            print(f"[{m}] 로드 {load_ms / 1000:.1f}s · {len(items)}문항", flush=True)
            started = time.perf_counter()
            rows = []
            for n, item in enumerate(items, start=1):
                result = answer_question(item.question, searcher.search, llm, definitions=definitions)
                rows.append(grade(item, result))
                if n % 10 == 0:
                    print(f"  {n}/{len(items)}", flush=True)
            runs[m] = rows
            s = summarize(rows)
            print(f"  완료 {time.perf_counter() - started:.0f}s · 거절 {_fmt(s['refusal_accuracy'])} · "
                  f"오거절 {_fmt(s['false_refusal_rate'])} · 인용적중 {_fmt(s['citation_hit'])} · "
                  f"수치 {_fmt(s['numeric_exact'])} · p50 {_fmt(s['llm_p50_ms'], False)}ms", flush=True)
            llm.unload()
    finally:
        searcher.close()

    note = (f"검색 dense(BGE-M3) top-8 → 근거 최대 5조항 · 6,000자 · 정의 조항 첨부 · num_ctx {OllamaChat().num_ctx} · "
            "temperature 0.1 · thinking off")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{args.name}.md").write_text(build_report(args.name, items, runs, {"note": note}), encoding="utf-8")
    (REPORT_DIR / f"{args.name}_review.md").write_text(build_review(args.name, items, runs), encoding="utf-8")
    (REPORT_DIR / f"{args.name}.json").write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ eval/reports/{args.name}.md, {args.name}_review.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
