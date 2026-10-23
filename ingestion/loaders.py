"""규정 문서(PDF / HWP / HWPX / TXT)를 페이지 단위 텍스트로 읽는다.

표는 가능한 경우 마크다운 표로 변환해 보존한다 (PRD F-ING-04).
HWP / HWPX 는 페이지 정보가 없으므로 섹션 번호를 페이지 번호 대신 사용한다.
"""
from __future__ import annotations

import re
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

SUPPORTED_EXTENSIONS = {".pdf", ".hwp", ".hwpx", ".txt"}


@dataclass
class Page:
    number: int
    text: str
    images: int = 0  # 쪽에 포함된 이미지 수 (별표가 이미지로 들어간 경우 감지용)


def load_document(path: str | Path, extract_tables: bool = True) -> list[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path, extract_tables)
    if ext == ".hwpx":
        return _load_hwpx(path)
    if ext == ".hwp":
        with open(path, "rb") as f:
            head = f.read(512)
        # 국가법령정보센터의 본문 "한글 저장"은 확장자만 .hwp 인 HWPML(XML) 파일이다
        if head.lstrip(b"\xef\xbb\xbf").startswith(b"<?xml") or b"<HWPML" in head:
            return _load_hwpml(path)
        return _load_hwp(path)
    if ext == ".txt":
        return [Page(1, path.read_text(encoding="utf-8"))]
    raise ValueError(f"지원하지 않는 형식: {path.name} (지원: {sorted(SUPPORTED_EXTENSIONS)})")


def table_to_markdown(rows: list[list[str]]) -> str:
    rows = [[_clean_cell(c) for c in row] for row in rows if any(c and c.strip() for c in row)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


CHECK_MARKS = {"○", "●", "◎", "√", "✓", "V", "v", "O", "o"}


def grid_to_blocks(cells: dict[tuple[int, int], list[str]], spans: dict[tuple[int, int], int],
                   rowspans: dict[tuple[int, int], int] | None = None) -> list[str]:
    """(행, 열) 주소로 모은 셀을 마크다운 표(또는 체크 표는 문장 목록)로 만든다.

    - 표 맨 위/아래에서 한 칸이 모든 열을 덮는 행(예: "비고 :")은 표 밖 문단으로 뺀다.
    - 두 줄 머리(1행에 가로 병합 칸, 예: "전결권자" 아래 "부장 · 부서장")는 열 이름을 "전결권자·부서장" 으로 합친다.
    - 세로 병합 칸의 값을 아래 행에도 채운다 — 표가 조각으로 나뉘어도 행마다 무슨 업무인지 알 수 있게.
    - ○ 표시로 담당자를 나타내는 체크 표(위임전결표 등)는 행마다 "업무 > 세부 → 전결권자·부서장" 문장으로 바꾼다.
      마크다운으로 두면 LLM 이 빈 열이 많은 표에서 ○ 의 열 위치를 세어야 한다.
    """
    if not cells:
        return []
    rowspans = rowspans or {}
    n_rows = max(r for r, _ in cells) + 1
    n_cols = max(max(c for _, c in cells) + 1, max(spans.values(), default=1))

    def full_width(r: int) -> bool:
        return n_cols > 1 and spans.get((r, 0), 1) >= n_cols

    def text(r: int, c: int) -> str:
        return " ".join(cells.get((r, c), [])).strip()

    top, bottom = 0, n_rows
    while top < bottom and full_width(top):
        top += 1
    while bottom > top and full_width(bottom - 1):
        bottom -= 1
    before = [text(r, 0) for r in range(0, top)]
    after = [text(r, 0) for r in range(bottom, n_rows)]

    header_rows = _header_depth(cells, spans, top, bottom, n_cols)
    names = _column_names(cells, spans, top, header_rows, n_cols)
    grid = _fill_row_spans(cells, rowspans, top + header_rows, bottom, n_cols)
    data = []
    for r in range(top + header_rows, bottom):
        data.extend(_explode_row([grid.get((r, c), []) for c in range(n_cols)]))

    if header_rows and _is_check_table(data):
        body = _check_rows_to_lines(names, data)
    else:
        rows = ([names] if header_rows else []) + data
        md = table_to_markdown(rows) if rows else ""
        body = [md] if md else []
    return [b for b in before + body + after if b.strip()]


def _header_depth(cells, spans, top: int, bottom: int, n_cols: int) -> int:
    """머리 행 수: 1행에 가로 병합 칸이 있고 그 아래 행이 그 칸들을 나눠 이름을 주면 2, 아니면 1."""
    if bottom - top < 2:
        return 1 if bottom - top >= 1 else 0
    grouped = [c for (r, c), s in spans.items() if r == top and s > 1 and " ".join(cells.get((r, c), [])).strip()]
    if not grouped or bottom - top < 3:
        return 1
    below = [c for (r, c) in cells if r == top + 1 and " ".join(cells.get((r, c), [])).strip()]
    covered = any(g <= c < g + spans[(top, g)] for g in grouped for c in below)
    return 2 if covered else 1


def _column_names(cells, spans, top: int, header_rows: int, n_cols: int) -> list[str]:
    levels = []
    for r in range(top, top + header_rows):
        label = [""] * n_cols
        for (rr, c), texts in cells.items():
            if rr != r:
                continue
            t = re.sub(r"\s+", " ", " ".join(texts)).strip()
            for cc in range(c, min(n_cols, c + max(1, spans.get((rr, c), 1)))):
                label[cc] = t
        levels.append(label)
    names = []
    for c in range(n_cols):
        parts = []
        for level in levels:
            t = level[c]
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        names.append("·".join(parts))
    return names


def _fill_row_spans(cells, rowspans, start: int, bottom: int, n_cols: int) -> dict[tuple[int, int], list[str]]:
    grid = {k: [t for t in v if t.strip()] for k, v in cells.items()}
    for (r, c), k in rowspans.items():
        if r < start or k <= 1 or not grid.get((r, c)):
            continue
        for rr in range(r + 1, min(bottom, r + k)):
            if not grid.get((rr, c)):
                grid[(rr, c)] = list(grid[(r, c)])
    return grid


def _explode_row(row: list[list[str]]) -> list[list[str]]:
    """한 행의 칸들이 모두 같은 수(2 이상)의 줄을 가지면 줄마다 행으로 펼친다.

    예: | 1급 / 2급 / 3급 | 105,639,000 / 92,598,000 / 80,541,000 | → 3행 (보수규정 별표7: 직급과 금액이 한 칸에 줄바꿈으로 들어 있다)
    """
    filled = [cell for cell in row if cell]
    lengths = {len(cell) for cell in filled}
    if len(filled) >= 2 and len(lengths) == 1 and (n := lengths.pop()) > 1:
        return [[cell[i] if cell else "" for cell in row] for i in range(n)]
    return [[" ".join(cell) for cell in row]]


def _is_check_table(data: list[list[str]]) -> bool:
    marked = sum(1 for row in data if any(cell in CHECK_MARKS for cell in row))
    return len(data) >= 2 and marked / len(data) >= 0.5


def _check_rows_to_lines(names: list[str], data: list[list[str]]) -> list[str]:
    lines = []
    for row in data:
        content, assigned = [], []
        for c, cell in enumerate(row):
            if not cell:
                continue
            if cell in CHECK_MARKS:
                assigned.append(names[c] or f"열{c + 1}")
            elif not content or content[-1] != cell:
                content.append(re.sub(r"\s+", " ", cell))
        if not content and not assigned:
            continue
        line = "- " + " > ".join(content)
        if assigned:
            line += " → " + ", ".join(assigned)
        lines.append(line)
    return ["\n".join(lines)] if lines else []


def _clean_cell(cell: str | None) -> str:
    if not cell:
        return ""
    return re.sub(r"\s+", " ", cell).replace("|", "/").strip()


# ---------------------------------------------------------------- PDF

def _load_pdf(path: Path, extract_tables: bool) -> list[Page]:
    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = _pdf_page_with_tables(page) if extract_tables else page.get_text("text", sort=True)
            pages.append(Page(i, text, images=len(page.get_images())))
    return pages


def _pdf_page_with_tables(page) -> str:
    """표 영역은 마크다운 표로, 나머지는 텍스트 블록으로 읽어 위→아래 순서로 합친다."""
    try:
        tables = list(page.find_tables().tables)
    except Exception:
        tables = []
    if not tables:
        return page.get_text("text", sort=True)

    import pymupdf

    rects = [pymupdf.Rect(t.bbox) for t in tables]
    items: list[tuple[float, float, str]] = []
    for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks", sort=True):
        if block_type != 0:
            continue
        block = pymupdf.Rect(x0, y0, x1, y1)
        if any(block.intersects(r) for r in rects):
            continue
        items.append((y0, x0, text))
    for table, rect in zip(tables, rects):
        md = table_to_markdown(table.extract())
        if md:
            items.append((rect.y0, rect.x0, md))
    items.sort(key=lambda it: (round(it[0]), it[1]))
    return "\n".join(text.strip("\n") for _, _, text in items)


# ---------------------------------------------------------------- HWPX (OWPML, zip + xml)

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _load_hwpx(path: Path) -> list[Page]:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if re.fullmatch(r"Contents/section\d+\.xml", n)]
        names.sort(key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[-1]).group(1)))
        pages = []
        for i, name in enumerate(names, start=1):
            root = ET.fromstring(zf.read(name))
            blocks: list[str] = []
            for child in root:
                if _local(child.tag) == "p":
                    blocks.extend(_hwpx_paragraph(child))
            pages.append(Page(i, "\n".join(blocks)))
    return pages


def _hwpx_paragraph(p) -> list[str]:
    blocks: list[str] = []
    buf: list[str] = []

    def visit(node):
        for child in node:
            name = _local(child.tag)
            if name == "t":
                buf.append("".join(child.itertext()))
            elif name == "tbl":
                if buf:
                    blocks.append("".join(buf))
                    buf.clear()
                blocks.append(_hwpx_table(child))
            elif name == "lineBreak":
                buf.append("\n")
            elif name == "tab":
                buf.append(" ")
            else:
                visit(child)

    visit(p)
    if buf:
        blocks.append("".join(buf))
    return [b for b in blocks if b.strip()]


def _hwpx_table(tbl) -> str:
    rows = []
    for tr in tbl:
        if _local(tr.tag) != "tr":
            continue
        cells = []
        for tc in tr:
            if _local(tc.tag) != "tc":
                continue
            cells.append(" ".join("".join(t.itertext()) for t in tc.iter() if _local(t.tag) == "t"))
        rows.append(cells)
    return table_to_markdown(rows)


# ---------------------------------------------------------------- HWPML (한글 XML, 단일 파일)

_HWPML_SKIP = {"HEADER", "FOOTER", "PICTURE", "SECDEF", "COLDEF", "AUTONUM", "FOOTNOTE", "ENDNOTE", "SHAPEOBJECT"}


def _load_hwpml(path: Path) -> list[Page]:
    raw = path.read_bytes().decode("utf-8-sig")
    # ElementTree 는 DTD 내부 엔티티를 지원하지 않으므로 DOCTYPE 을 걷어내고 &nbsp; 를 직접 치환한다
    raw = re.sub(r"<!DOCTYPE[^\[>]*(\[.*?\])?\s*>", "", raw, count=1, flags=re.S).replace("&nbsp;", "&#160;")
    # 국가법령정보센터 HWPML 은 본문의 "<2014. 12. 15.>", "<삭제 …>" 같은 문자를 이스케이프하지 않는다
    raw = re.sub(r"<(?![A-Za-z/!?])", "&lt;", raw)
    raw = re.sub(r"&(?!#\d+;|#x[0-9A-Fa-f]+;|[A-Za-z]\w*;)", "&amp;", raw)
    root = ET.fromstring(raw)
    body = root.find("BODY")
    sections = body.findall("SECTION") if body is not None else []
    pages = []
    for i, section in enumerate(sections, start=1):
        blocks: list[str] = []
        for p in section.findall("P"):
            blocks.extend(_hwpml_paragraph(p))
        pages.append(Page(i, "\n".join(blocks)))
    return pages


def _hwpml_paragraph(p) -> list[str]:
    blocks: list[str] = []
    buf: list[str] = []
    for text in p.findall("TEXT"):
        for child in text:
            if child.tag == "CHAR":
                buf.append("".join(child.itertext()))
            elif child.tag == "TABLE":
                if "".join(buf).strip():
                    blocks.append("".join(buf))
                buf.clear()
                blocks.extend(_hwpml_table(child))
            elif child.tag not in _HWPML_SKIP:
                buf.append("".join(child.itertext()))
    if "".join(buf).strip():
        blocks.append("".join(buf))
    return blocks


def _hwpml_table(tbl) -> list[str]:
    cells: dict[tuple[int, int], list[str]] = {}
    spans: dict[tuple[int, int], int] = {}
    rowspans: dict[tuple[int, int], int] = {}
    for cell in tbl.findall("ROW/CELL"):  # 직계 셀만 (중첩 표는 셀 텍스트로 합쳐짐)
        key = (int(cell.get("RowAddr", 0)), int(cell.get("ColAddr", 0)))
        spans[key] = int(cell.get("ColSpan", 1))
        rowspans[key] = int(cell.get("RowSpan", 1))
        paralist = cell.find("PARALIST")
        texts = []
        for p in (paralist.findall("P") if paralist is not None else []):
            texts.extend(b.strip() for b in _hwpml_paragraph(p) if b.strip())
        cells[key] = texts
    return grid_to_blocks(cells, spans, rowspans)


# ---------------------------------------------------------------- HWP 5.0 (OLE 바이너리)

HWPTAG_PARA_HEADER = 66
HWPTAG_PARA_TEXT = 67
HWPTAG_CTRL_HEADER = 71
HWPTAG_LIST_HEADER = 72
HWPTAG_TABLE = 77
# 8 wchar 를 차지하는 확장 · 인라인 컨트롤 문자 (HWP 5.0 문서 형식 명세)
_WIDE_CONTROLS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def _load_hwp(path: Path) -> list[Page]:
    import olefile

    with olefile.OleFileIO(str(path)) as ole:
        header = ole.openstream("FileHeader").read()
        flags = struct.unpack("<I", header[36:40])[0]
        if flags & 0x2:
            raise ValueError(f"암호가 걸린 HWP 문서입니다: {path.name}")
        if flags & 0x4:
            raise ValueError(f"배포용 HWP 문서는 본문을 읽을 수 없습니다. PDF로 변환해 주세요: {path.name}")
        compressed = bool(flags & 0x1)
        sections = [e for e in ole.listdir() if len(e) == 2 and e[0] == "BodyText"]
        sections.sort(key=lambda e: int(re.sub(r"\D", "", e[1]) or 0))
        pages = []
        for i, entry in enumerate(sections, start=1):
            data = ole.openstream(entry).read()
            if compressed:
                data = zlib.decompress(data, -15)
            pages.append(Page(i, hwp_section_text(data)))
    return pages


def hwp_section_text(data: bytes) -> str:
    """본문 문단은 줄 단위로, 표는 마크다운 표로 복원한다.

    표 구조: CTRL_HEADER('tbl ', 레벨 L) → TABLE(L+1) → 셀마다 LIST_HEADER(L+1) + 셀 문단들.
    LIST_HEADER 의 8바이트 이후에 열 · 행 · 열병합 · 행병합(각 UINT16)이 온다.
    중첩 표의 셀은 바깥 셀의 텍스트로 합친다.
    """
    blocks: list[str] = []
    table: dict | None = None  # {"level", "cells": {(row, col): [text]}, "spans": {(row, col): 열병합}, ...}

    def flush_table():
        nonlocal table
        if table:
            blocks.extend(grid_to_blocks(table["cells"], table["spans"], table["rowspans"]))
        table = None

    for tag, level, payload in iter_hwp_records(data):
        if table is not None and level <= table["level"]:
            flush_table()
        if tag == HWPTAG_CTRL_HEADER and payload[:4] == b" lbt" and table is None:
            table = {"level": level, "cells": {}, "spans": {}, "rowspans": {}, "cell_level": level + 1, "current": None}
        elif tag == HWPTAG_LIST_HEADER and table is not None and level == table["cell_level"] and len(payload) >= 16:
            col, row, col_span, row_span = struct.unpack_from("<HHHH", payload, 8)
            table["current"] = (row, col)
            table["cells"].setdefault((row, col), [])
            table["spans"][(row, col)] = col_span
            table["rowspans"][(row, col)] = row_span
        elif tag == HWPTAG_PARA_TEXT:
            text = decode_hwp_para_text(payload).strip()
            if not text:
                continue
            if table is not None and table["current"] is not None:
                table["cells"][table["current"]].append(text)
            elif table is None:
                blocks.append(text)
    flush_table()
    return "\n".join(blocks)


def iter_hwp_records(data: bytes):
    pos, end = 0, len(data)
    while pos + 4 <= end:
        (header,) = struct.unpack_from("<I", data, pos)
        pos += 4
        tag = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        if size == 0xFFF:
            (size,) = struct.unpack_from("<I", data, pos)
            pos += 4
        yield tag, level, data[pos:pos + size]
        pos += size


def decode_hwp_para_text(payload: bytes) -> str:
    count = len(payload) // 2
    units = struct.unpack(f"<{count}H", payload[: count * 2])
    out: list[int] = []
    i = 0
    while i < count:
        code = units[i]
        if code < 32:
            if code == 10:
                out.append(ord("\n"))
            elif code == 9:
                out.append(ord(" "))
            i += 8 if code in _WIDE_CONTROLS else 1
            continue
        out.append(code)
        i += 1
    return struct.pack(f"<{len(out)}H", *out).decode("utf-16-le", errors="ignore")
