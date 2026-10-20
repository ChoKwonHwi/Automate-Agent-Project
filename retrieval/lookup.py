"""조항 참조로 원문 청크 조회 (Golden Set 작성 · 근거 확인용, 벡터 인덱스 불필요).

    python -m retrieval.lookup "여비규정 제16조"
    python -m retrieval.lookup "여비규정 별표1" "임직원행동강령 제17조"
    python -m retrieval.lookup --list 여비규정        # 해당 규정의 참조 목록
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict

from .config import get_settings
from .index import load_chunks
from .refs import chunk_ref, normalize_ref


def group_by_ref(chunks: list[dict]) -> "OrderedDict[str, list[dict]]":
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for c in chunks:
        groups.setdefault(normalize_ref(chunk_ref(c)), []).append(c)
    return groups


def lookup(chunks: list[dict], ref: str) -> list[dict]:
    return group_by_ref(chunks).get(normalize_ref(ref), [])


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="조항 참조로 원문 조회")
    parser.add_argument("refs", nargs="*", help='예: "여비규정 제16조", "여비규정 별표1"')
    parser.add_argument("--list", metavar="규정명", help="규정의 참조 목록 출력")
    args = parser.parse_args(argv)

    chunks = load_chunks(get_settings().chunks_path)
    if args.list:
        prefix = normalize_ref(args.list)
        refs = list(OrderedDict.fromkeys(chunk_ref(c) for c in chunks if normalize_ref(chunk_ref(c)).startswith(prefix)))
        titles = {chunk_ref(c): c.get("article_title", "") for c in chunks}
        for r in refs:
            print(f"{r}" + (f"({titles[r]})" if titles.get(r) else ""))
        return 0 if refs else 1

    status = 0
    for ref in args.refs:
        found = lookup(chunks, ref)
        print(f"===== {ref}" + ("" if found else "  → 찾을 수 없음 (--list 로 참조 표기를 확인하세요)"))
        if not found:
            status = 1
            continue
        c0 = found[0]
        print(f"[{c0['org']}] {c0['doc_title']} · 시행 {c0['effective_date']} · {c0['section']}")
        for c in found:
            print(c["text"])
        print()
    return status


if __name__ == "__main__":
    sys.exit(main())
