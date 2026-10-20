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
    articles = [c for c in result.chunks if c["section"] == "본칙" and c["part"] == 0]
    if result.pages and result.chars < 100 * result.pages:
        warnings.append("텍스트가 거의 없음 — 스캔본(이미지 PDF)일 수 있음, OCR 필요")
    if not articles:
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


def build_report(results: list[DocResult], unregistered: list[str], sample_size: int = 20, seed: int = 42) -> str:
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

    articles = [c for r in ok for c in r.chunks if c["section"] in ("본칙", "별표")]
    if articles:
        rng = random.Random(seed)
        sample = rng.sample(articles, min(sample_size, len(articles)))
        out += ["## 수작업 검수 샘플", "",
                "각 청크가 **하나의 조(또는 별표)만 온전히** 담고 있는지 확인하고 체크한다.",
                "경계 오류 = 앞/뒤 조 내용이 섞였거나, 조문 중간에서 잘렸거나(분할 표시 없이), 조 제목이 틀린 경우.", ""]
        for c in sample:
            preview = c["text"][:400] + (" …" if len(c["text"]) > 400 else "")
            out += [f"- [ ] `{c['chunk_id']}` — {c['doc_title']} {c['article_label']} "
                    f"(p.{c['page_start']}–{c['page_end']}, part {c['part'] + 1}/{c['total_parts']})",
                    "", "```text", preview, "```", ""]
        out += ["경계 오류율 = 오류 개수 / 샘플 수 → **5% 미만**이면 W1 완료.", ""]

    return "\n".join(out)
