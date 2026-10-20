"""파싱 품질 검수 리포트 (data/processed/report.md).

W1 완료 기준(조항 경계 오류 < 5%)을 확인하기 위해
문서별 통계 · 누락 조번호 · 경고와 수작업 검수용 무작위 샘플을 만든다.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime

from .sources import Source


@dataclass
class DocResult:
    source: Source
    pages: int = 0
    chars: int = 0
    chunks: list[dict] = field(default_factory=list)
    image_pages: list[int] = field(default_factory=list)
    error: str = ""


def missing_article_numbers(chunks: list[dict]) -> list[int]:
    numbers = {
        int(m.group(1))
        for c in chunks
        if c["section"] == "본칙" and (m := re.match(r"제(\d+)조", c["article_label"]))
    }
    if not numbers:
        return []
    return sorted(set(range(1, max(numbers) + 1)) - numbers)


def doc_warnings(result: DocResult) -> list[str]:
    warnings = []
    if not result.source.source_url.startswith("http"):
        warnings.append("sources.csv 의 source_url 이 비어 있음(TODO) — 출처 URL 입력 필요")
    articles = [c for c in result.chunks if c["section"] == "본칙" and c["part"] == 0]
    if result.pages and result.chars < 100 * result.pages:
        warnings.append("텍스트가 거의 없음 — 스캔본(이미지 PDF)일 수 있음, OCR 필요")
    is_annex = any(c["section"] == "별표" for c in result.chunks)
    if not articles and not is_annex:
        warnings.append("조(條)를 하나도 인식하지 못함 — 원문 형식 확인 필요")
    if result.image_pages:
        warnings.append(f"글자 없이 이미지만 있는 쪽 {result.image_pages} — 별표(금액표)가 이미지일 수 있음. "
                        "별표 원본 파일을 따로 받거나 OCR / 수기 입력 필요")
    missing = missing_article_numbers(result.chunks)
    if missing:
        warnings.append(f"누락 조번호 {missing[:15]}{' …' if len(missing) > 15 else ''} — 경계 인식 실패 가능")
    preamble = [c for c in result.chunks if c["section"] == "전문"]
    if preamble and preamble[0]["char_len"] > 1500:
        warnings.append(f"전문(제1조 이전) 길이 {preamble[0]['char_len']}자 — 조 헤딩 형식이 다를 수 있음")
    return warnings


def build_report(results: list[DocResult], unregistered: list[str], sample_size: int = 30, seed: int = 42) -> str:
    out = [f"# 파싱 검수 리포트", "", f"생성: {datetime.now():%Y-%m-%d %H:%M}", ""]

    ok = [r for r in results if not r.error]
    total_chunks = sum(len(r.chunks) for r in ok)
    out += [f"- 문서 {len(results)}개 (성공 {len(ok)}, 실패 {len(results) - len(ok)})", f"- 청크 {total_chunks}개", ""]

    out += ["## 문서별 통계", "",
            "| doc_id | 페이지 | 본칙 조 | 부칙 | 별표 | 청크 | 최장 청크(자) | 상태 |",
            "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.error:
            out.append(f"| {r.source.doc_id} | - | - | - | - | - | - | ❌ {r.error} |")
            continue
        heads = [c for c in r.chunks if c["part"] == 0]
        n_article = sum(1 for c in heads if c["section"] == "본칙")
        n_supp = len({c["article_label"].split()[0] for c in heads if c["section"] == "부칙"})
        n_annex = sum(1 for c in heads if c["section"] == "별표")
        longest = max((c["char_len"] for c in r.chunks), default=0)
        status = "⚠️ 경고" if doc_warnings(r) else "✅"
        out.append(f"| {r.source.doc_id} | {r.pages} | {n_article} | {n_supp} | {n_annex} | "
                   f"{len(r.chunks)} | {longest} | {status} |")
    out.append("")

    warned = [(r, doc_warnings(r)) for r in ok]
    warned = [(r, w) for r, w in warned if w]
    if warned:
        out += ["## 경고", ""]
        for r, warnings in warned:
            out += [f"**{r.source.doc_id}**"] + [f"- {w}" for w in warnings] + [""]

    if unregistered:
        out += ["## sources.csv 에 등록되지 않은 파일", ""] + [f"- `{p}`" for p in unregistered] + [""]

    sample = review_sample(ok, sample_size, seed)
    if sample:
        out += ["## 수작업 검수 샘플", "",
                f"본칙 · 별표 · 부칙에서 문서별로 고르게 {len(sample)}개를 뽑았다. 각 항목에서 확인할 것:",
                "1. **시작**: 청크 첫 줄이 해당 조(별표) 헤딩인가? (분할 조각은 `(계속)` 머리 다음 줄)",
                "2. **끝**: 마지막 줄 다음에 오는 `▼ 다음 청크 첫 줄` 이 바로 다음 조(별표) 헤딩인가? — 아니면 조문 중간에서 잘린 것",
                "3. **섞임**: 본문 안에 다른 조의 헤딩(`제N조(...)`)이 들어가 있지 않은가?",
                "4. **메타데이터**: 제목 줄의 조번호 · 조제목이 본문과 일치하는가? (별표는 표의 숫자가 맞는 열에 있는가)",
                "",
                "문제가 없으면 `[ ]` 를 `[x]` 로 바꾸고, 오류면 `[!]` 로 바꾼 뒤 아래에 사유를 한 줄 적는다.",
                "판단이 어려우면 `파일` 경로의 원문을 열어 조번호로 검색해 대조한다.",
                "",
                "> 이 파일은 `python -m ingestion.run` 을 돌릴 때마다 다시 만들어진다. "
                "체크는 `data/processed/review_w1.md` 로 **복사한 사본**에 한다.", ""]
        for item in sample:
            c = item["chunk"]
            out += [f"- [ ] `{c['chunk_id']}` — **{c['doc_title']} {c['article_label']}**"
                    + (f"({c['article_title']})" if c["article_title"] else "")
                    + f" · {c['section']} · part {c['part'] + 1}/{c['total_parts']} · 파일 `{item['file']}`",
                    "", "```text",
                    f"▲ 이전 청크 마지막 줄: {item['prev'] or '(문서 시작)'}",
                    "────────",
                    c["text"],
                    "────────",
                    f"▼ 다음 청크 첫 줄: {item['next'] or '(문서 끝)'}",
                    "```", ""]
        out += [f"경계 오류율 = `[!]` 개수 / {len(sample)} → **5% 미만**(오류 1개 이하)이면 W1 완료.", ""]

    return "\n".join(out)


def review_sample(results: list[DocResult], size: int = 30, seed: int = 42) -> list[dict]:
    """검수용 층화 표본: 본칙 70% · 별표 20% · 부칙 10%, 문서별로 돌아가며 고른다."""
    rng = random.Random(seed)
    quotas = {"본칙": round(size * 0.7), "별표": round(size * 0.2)}
    quotas["부칙"] = size - sum(quotas.values())

    def context(chunks: list[dict], i: int) -> dict:
        prev_text = chunks[i - 1]["text"].splitlines()[-1] if i > 0 else ""
        next_text = chunks[i + 1]["text"].splitlines()[0] if i + 1 < len(chunks) else ""
        return {"prev": prev_text[:120], "next": next_text[:120]}

    picked: list[dict] = []
    for section, quota in quotas.items():
        pools = []
        for r in results:
            idx = [i for i, c in enumerate(r.chunks) if c["section"] == section]
            rng.shuffle(idx)
            if idx:
                pools.append((r, idx))
        rng.shuffle(pools)
        taken = 0
        while taken < quota and any(idx for _, idx in pools):
            for r, idx in pools:
                if taken >= quota:
                    break
                if idx:
                    i = idx.pop()
                    picked.append({"chunk": r.chunks[i], "file": r.source.file_path, **context(r.chunks, i)})
                    taken += 1
    return picked
