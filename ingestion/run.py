"""규정 문서 일괄 파싱.

    python -m ingestion.run [--max-chars 1200] [--no-tables]

data/sources.csv 에 등록된 문서를 읽어
data/processed/chunks.jsonl 과 data/processed/report.md 를 만든다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chunker import chunk_document
from .clean import clean_pages
from .loaders import SUPPORTED_EXTENSIONS, load_document
from .report import DocResult, build_report
from .sources import load_sources

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="규정 문서를 조항 단위 청크로 변환")
    parser.add_argument("--sources", default="data/sources.csv")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--max-chars", type=int, default=1200, help="청크 최대 글자 수")
    parser.add_argument("--no-tables", action="store_true", help="PDF 표 추출 끄기 (속도 우선)")
    args = parser.parse_args(argv)

    sources = load_sources(ROOT / args.sources)
    if not sources:
        print(f"{args.sources} 에 등록된 문서가 없습니다. docs/data-collection.md 를 참고해 등록해 주세요.")
        return 1

    results: list[DocResult] = []
    for src in sources:
        path = ROOT / src.file_path
        print(f"- {src.doc_id} ({src.file_path}) ... ", end="", flush=True)
        if not path.exists():
            results.append(DocResult(src, error="파일 없음"))
            print("파일 없음")
            continue
        try:
            pages = load_document(path, extract_tables=not args.no_tables)
            lines = clean_pages(pages)
            chunks = chunk_document(lines, src.meta, max_chars=args.max_chars)
        except Exception as exc:  # 문서 하나가 실패해도 나머지는 계속 처리
            results.append(DocResult(src, error=f"{type(exc).__name__}: {exc}"))
            print(f"실패 — {exc}")
            continue
        results.append(DocResult(src, pages=len(pages), chars=sum(len(l) for l, _ in lines), chunks=chunks))
        print(f"{len(pages)}쪽 → 청크 {len(chunks)}개")

    registered = {(ROOT / s.file_path).resolve() for s in sources}
    raw_dir = ROOT / args.raw_dir
    unregistered = sorted(
        p.relative_to(ROOT).as_posix()
        for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and p.resolve() not in registered
    )

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            for chunk in r.chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (out_dir / "report.md").write_text(build_report(results, unregistered), encoding="utf-8")

    failed = sum(1 for r in results if r.error)
    print(f"\n완료: 청크 {sum(len(r.chunks) for r in results)}개, 실패 {failed}개")
    print(f"→ {(out_dir / 'chunks.jsonl').relative_to(ROOT)}, {(out_dir / 'report.md').relative_to(ROOT)}")
    if unregistered:
        print(f"주의: sources.csv 에 없는 파일 {len(unregistered)}개 (report.md 참고)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
