"""검색 평가: Golden Set 으로 Recall@k · MRR 측정.

    python -m eval.retrieval [--k 10] [--name w2_dense]

정답 판정은 chunk_id 가 아니라 조항 참조("여비규정 제16조") 단위 — 분할 조각 중 하나만 걸려도 정답.
결과: eval/reports/{name}.md (요약 · 유형별 · 분류별 · 실패 문항), eval/reports/{name}.json (문항별 원자료)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime

from retrieval.config import ROOT
from retrieval.refs import normalize_ref

from .golden import GoldenItem, load_golden

REPORT_DIR = ROOT / "eval/reports"
CUTOFFS = (1, 3, 5, 10)


def first_hit_rank(retrieved_refs: list[str], evidence: list[str]) -> int | None:
    """정답 조항이 처음 나온 순위 (1부터). 없으면 None."""
    gold = {normalize_ref(r) for r in evidence}
    for rank, ref in enumerate(retrieved_refs, start=1):
        if normalize_ref(ref) in gold:
            return rank
    return None


def coverage(retrieved_refs: list[str], evidence: list[str], k: int) -> float:
    """상위 k 안에 들어온 근거 조항 비율 (multi 문항에서 의미 있음)."""
    gold = {normalize_ref(r) for r in evidence}
    found = {normalize_ref(r) for r in retrieved_refs[:k]} & gold
    return len(found) / len(gold) if gold else 0.0


def dedupe_refs(refs: list[str]) -> list[str]:
    """같은 조항의 분할 조각이 여러 개 걸리면 첫 번째만 남긴다 (순위 = 조항 단위)."""
    seen, out = set(), []
    for r in refs:
        if normalize_ref(r) not in seen:
            seen.add(normalize_ref(r))
            out.append(r)
    return out


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {"n": 0}
    out = {"n": n}
    for k in CUTOFFS:
        out[f"recall@{k}"] = sum(1 for r in results if r["rank"] and r["rank"] <= k) / n
    out["mrr"] = sum(1 / r["rank"] for r in results if r["rank"]) / n
    out["coverage@5"] = sum(r["coverage@5"] for r in results) / n
    return out


def evaluate(items: list[GoldenItem], search, k: int) -> list[dict]:
    """search(question, k) → [(score, ref, chunk)] 를 받아 문항별 결과를 만든다."""
    results = []
    for item in items:
        started = time.perf_counter()
        hits = search(item.question, k * 3)  # 분할 조각 중복 제거 후에도 k개가 남도록 넉넉히
        latency = (time.perf_counter() - started) * 1000
        refs = dedupe_refs([ref for _, ref, _ in hits])[:k]
        scores = {}
        for score, ref, _ in hits:
            scores.setdefault(normalize_ref(ref), score)
        results.append({
            "id": item.id, "type": item.type, "category": item.category, "question": item.question,
            "evidence": item.evidence, "retrieved": refs,
            "scores": [round(scores[normalize_ref(r)], 4) for r in refs],
            "rank": first_hit_rank(refs, item.evidence) if item.evidence else None,
            "coverage@5": coverage(refs, item.evidence, 5) if item.evidence else 0.0,
            "latency_ms": round(latency, 1),
        })
    return results


def _metric_row(label: str, s: dict) -> str:
    if not s.get("n"):
        return f"| {label} | 0 | - | - | - | - | - |"
    return (f"| {label} | {s['n']} | {s['recall@1']:.2f} | {s['recall@3']:.2f} | {s['recall@5']:.2f} | "
            f"{s['recall@10']:.2f} | {s['mrr']:.3f} |")


def build_report(name: str, results: list[dict], settings_note: str) -> str:
    answerable = [r for r in results if r["type"] != "refuse"]
    refuse = [r for r in results if r["type"] == "refuse"]
    overall = summarize(answerable)
    out = [f"# 검색 평가 — {name}", "", f"- 생성: {datetime.now():%Y-%m-%d %H:%M}", f"- 설정: {settings_note}",
           f"- 문항: 답변 가능 {len(answerable)} · 거절(refuse) {len(refuse)}", "",
           "정답 판정: 근거 조항(규정명 + 조번호) 중 하나라도 상위 k 안에 있으면 적중. 순위는 조항 단위(분할 조각 중복 제거).", "",
           "## 요약", "", "| 구분 | n | R@1 | R@3 | R@5 | R@10 | MRR |", "|---|---|---|---|---|---|---|",
           _metric_row("전체", overall)]

    by_type = defaultdict(list)
    for r in answerable:
        by_type[r["type"]].append(r)
    for t in ("fact", "numeric", "multi"):
        out.append(_metric_row(t, summarize(by_type.get(t, []))))
    multi = by_type.get("multi", [])
    if multi:
        out += ["", f"multi 문항 근거 커버리지@5 (근거 조항 중 상위 5에 들어온 비율 평균): "
                    f"**{summarize(multi)['coverage@5']:.2f}**"]

    by_cat = defaultdict(list)
    for r in answerable:
        by_cat[r["category"] or "-"].append(r)
    out += ["", "## 분류별", "", "| 분류 | n | R@1 | R@3 | R@5 | R@10 | MRR |", "|---|---|---|---|---|---|---|"]
    for cat, rs in sorted(by_cat.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append(_metric_row(cat, summarize(rs)))

    latencies = [r["latency_ms"] for r in results]
    out += ["", "## 검색 지연", "",
            f"- p50 {statistics.median(latencies):.0f}ms · p95 {sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]:.0f}ms "
            f"(질문 임베딩 + 벡터 검색, 모델 로드 제외)"]

    if refuse:
        top_ans = [r["scores"][0] for r in answerable if r["scores"]]
        top_ref = [r["scores"][0] for r in refuse if r["scores"]]
        out += ["", "## 거절 문항 참고 (W3 거절 임계값 설계용)", "",
                f"- 답변 가능 문항의 top-1 점수: 최소 {min(top_ans):.3f} · 중앙값 {statistics.median(top_ans):.3f}",
                f"- 거절 문항의 top-1 점수: 최대 {max(top_ref):.3f} · 중앙값 {statistics.median(top_ref):.3f}",
                "- 두 분포가 겹치면 점수 임계값만으로는 거절을 판단할 수 없다 → LLM 판정 · 리랭커 점수 병행 필요", ""]
        for r in refuse:
            out.append(f"  - `{r['id']}` {r['question']} → top-1 {r['retrieved'][0] if r['retrieved'] else '-'} "
                       f"({r['scores'][0] if r['scores'] else '-'})")

    misses = [r for r in answerable if not r["rank"] or r["rank"] > 5]
    out += ["", f"## 상위 5 안에 정답이 없는 문항 ({len(misses)})", ""]
    for r in misses:
        out += [f"- `{r['id']}` [{r['type']}/{r['category']}] {r['question']}",
                f"  - 정답: {'; '.join(r['evidence'])} → 순위 {r['rank'] or '10위 밖'}",
                f"  - 검색 상위 3: {' | '.join(r['retrieved'][:3])}"]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Golden Set 검색 평가")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--name", default="w2_dense_baseline", help="리포트 파일 이름")
    args = parser.parse_args(argv)

    from retrieval.search import Searcher

    items = load_golden()
    searcher = Searcher()
    try:
        searcher.embedder.model  # 모델 로드를 지연 측정에서 제외
        search = lambda q, k: [(h.score, h.ref, h.chunk) for h in searcher.search(q, k)]
        results = evaluate(items, search, args.k)
    finally:
        searcher.close()

    s = searcher.settings
    note = f"dense only · {s.embedding_model} · Qdrant cosine · 필터 없음 · k={args.k}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(args.name, results, note)
    (REPORT_DIR / f"{args.name}.md").write_text(report, encoding="utf-8")
    (REPORT_DIR / f"{args.name}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    overall = summarize([r for r in results if r["type"] != "refuse"])
    print(f"문항 {overall['n']}개 (refuse 제외) · R@1 {overall['recall@1']:.2f} · R@5 {overall['recall@5']:.2f} · "
          f"R@10 {overall['recall@10']:.2f} · MRR {overall['mrr']:.3f}")
    print(f"→ eval/reports/{args.name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
