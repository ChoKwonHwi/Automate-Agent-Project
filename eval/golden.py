"""Golden Set 읽기 · 검사.

    python -m eval.golden            # data/eval/golden_set.csv 형식 검사 + 구성 현황
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from retrieval.config import ROOT, get_settings
from retrieval.index import load_chunks
from retrieval.lookup import group_by_ref
from retrieval.refs import normalize_ref, split_refs

GOLDEN_PATH = ROOT / "data/eval/golden_set.csv"
COLUMNS = ["id", "type", "category", "question", "answer", "evidence", "notes"]
TYPES = {"fact": 50, "numeric": 20, "multi": 15, "refuse": 15}  # 목표 개수 (docs/plan-w2.md)


@dataclass
class GoldenItem:
    id: str
    type: str
    category: str
    question: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    notes: str = ""


def read_rows(path: Path) -> list[tuple[int, dict]]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"{path} 인코딩을 읽을 수 없습니다. 'CSV UTF-8' 로 저장해 주세요.")
    reader = csv.DictReader(io.StringIO(text))
    return [(i, row) for i, row in enumerate(reader, start=2) if any((v or "").strip() for k, v in row.items() if k)]


def check(rows: list[tuple[int, dict]], known_refs: set[str]) -> tuple[list[GoldenItem], list[str]]:
    items, problems = [], []
    seen: set[str] = set()
    for line, row in rows:
        where = f"{line}행({(row.get('id') or '?').strip()})"
        if None in row:  # 칸 수가 머리줄보다 많음 → 따옴표 없는 쉼표
            problems.append(f"{where}: 칸이 {len(COLUMNS)}개보다 많습니다. 쉼표가 들어간 칸은 큰따옴표로 감싸세요 "
                            f"(넘친 값: {row[None]})")
            continue
        r = {k: (row.get(k) or "").strip() for k in COLUMNS}
        refs = split_refs(r["evidence"])
        if not r["id"]:
            problems.append(f"{where}: id 가 비어 있습니다.")
        elif r["id"] in seen:
            problems.append(f"{where}: id 중복")
        seen.add(r["id"])
        if r["type"] not in TYPES:
            problems.append(f"{where}: type 은 {sorted(TYPES)} 중 하나여야 합니다 (현재: '{r['type']}')")
        if not r["question"]:
            problems.append(f"{where}: question 이 비어 있습니다.")
        if r["type"] == "refuse":
            if refs:
                problems.append(f"{where}: refuse 문항은 evidence 를 비워 둡니다.")
        else:
            if not r["answer"]:
                problems.append(f"{where}: answer 가 비어 있습니다.")
            if not refs:
                problems.append(f"{where}: evidence 가 비어 있습니다.")
            if r["type"] == "multi" and len(refs) < 2:
                problems.append(f"{where}: multi 문항은 근거 조항이 2개 이상이어야 합니다.")
            for ref in refs:
                if normalize_ref(ref) not in known_refs:
                    problems.append(f"{where}: 근거 '{ref}' 를 찾을 수 없습니다 — "
                                    "`python -m retrieval.lookup --list 규정명` 의 표기를 쓰세요 (별표는 '여비규정 별표1')")
        items.append(GoldenItem(r["id"], r["type"], r["category"], r["question"], r["answer"], refs, r["notes"]))
    return items, problems


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    known = set(group_by_ref(load_chunks(get_settings().chunks_path)))
    items, problems = check(read_rows(path), known)
    if problems:
        raise ValueError("Golden Set 형식 오류:\n" + "\n".join(problems))
    return items


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Golden Set 형식 검사")
    parser.add_argument("path", nargs="?", default=str(GOLDEN_PATH))
    args = parser.parse_args(argv)

    known = set(group_by_ref(load_chunks(get_settings().chunks_path)))
    items, problems = check(read_rows(Path(args.path)), known)
    print(f"문항 {len(items)}개")
    types = Counter(i.type for i in items)
    print("유형별: " + ", ".join(f"{t} {types.get(t, 0)}/{goal}" for t, goal in TYPES.items()))
    cats = Counter(i.category for i in items if i.type != "refuse")
    print("분류별: " + (", ".join(f"{c} {n}" for c, n in cats.most_common()) or "-"))
    doc_refs = Counter(normalize_ref(r) for i in items for r in i.evidence)
    overused = [r for r, n in doc_refs.items() if n >= 3]
    if overused:
        print(f"참고: 같은 조항을 3번 이상 근거로 쓴 문항이 있습니다 {overused}")
    if problems:
        print(f"\n문제 {len(problems)}건")
        for p in problems:
            print(" - " + p)
        return 1
    print("\n형식 문제 없음 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
