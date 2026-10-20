"""data/sources.csv 읽기. 엑셀에서 저장한 CP949 파일도 허용한다."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ["org", "org_code", "title", "category", "effective_date", "source_url", "collected_at", "file_path"]


@dataclass
class Source:
    org: str
    org_code: str
    title: str
    category: str
    effective_date: str
    source_url: str
    collected_at: str
    file_path: str
    doc_id: str
    notes: str = ""

    @property
    def meta(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "org": self.org,
            "title": self.title,
            "category": self.category,
            "effective_date": self.effective_date,
            "source_url": self.source_url,
        }


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path} 인코딩을 읽을 수 없습니다. 'CSV UTF-8' 형식으로 저장해 주세요.")


def load_sources(path: str | Path) -> list[Source]:
    path = Path(path)
    reader = csv.DictReader(io.StringIO(_read_text(path)))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"{path} 에 필수 컬럼이 없습니다: {missing}")

    sources, seen = [], set()
    for line_no, row in enumerate(reader, start=2):
        row = {k: (v or "").strip() for k, v in row.items() if k}
        if not any(row.values()):
            continue
        empty = [c for c in REQUIRED_COLUMNS if not row.get(c)]
        if empty:
            raise ValueError(f"{path}:{line_no} 필수 값이 비어 있습니다: {empty}")
        if row.get("doc_id", "").startswith("http"):
            raise ValueError(f"{path}:{line_no} doc_id 칸(9번째)에 URL이 들어 있습니다. "
                             "URL은 source_url 칸(6번째)에 넣고 doc_id 는 비워 두세요.")
        file_path = row["file_path"].replace("\\", "/")
        doc_id = row.get("doc_id") or f"{row['org_code']}-{Path(file_path).stem}"
        if doc_id in seen:
            raise ValueError(f"{path}:{line_no} doc_id 중복: {doc_id}")
        seen.add(doc_id)
        sources.append(Source(
            org=row["org"], org_code=row["org_code"], title=row["title"], category=row["category"],
            effective_date=row["effective_date"], source_url=row["source_url"],
            collected_at=row["collected_at"], file_path=file_path, doc_id=doc_id,
            notes=row.get("notes", ""),
        ))
    return sources
