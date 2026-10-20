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


def grid_to_blocks(cells: dict[tuple[int, int], list[str]], spans: dict[tuple[int, int], int]) -> list[str]:
    """(행, 열) 주소로 모은 셀을 마크다운 표로 만든다.

    표 맨 위/아래에서 한 칸이 모든 열을 덮는 행(예: "비고 :")은 표 밖 문단으로 뺀다.
    """
    if not cells:
        return []
    n_rows = max(r for r, _ in cells) + 1
    n_cols = max(max(c for _, c in cells) + 1, max(spans.values(), default=1))

    def full_width(r: int) -> bool:
        return n_cols > 1 and spans.get((r, 0), 1) >= n_cols

    def text(r: int, c: int) -> str:
        return " ".join(cells.get((r, c), []))

    top, bottom = 0, n_rows
    while top < bottom and full_width(top):
        top += 1
    while bottom > top and full_width(bottom - 1):
        bottom -= 1
    blocks = [text(r, 0) for r in range(0, top)]
    md = table_to_markdown([[text(r, c) for c in range(n_cols)] for r in range(top, bottom)])
    if md:
        blocks.append(md)
    blocks += [text(r, 0) for r in range(bottom, n_rows)]
    return [b for b in blocks if b.strip()]


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
    for cell in tbl.findall("ROW/CELL"):  # 직계 셀만 (중첩 표는 셀 텍스트로 합쳐짐)
        key = (int(cell.get("RowAddr", 0)), int(cell.get("ColAddr", 0)))
        spans[key] = int(cell.get("ColSpan", 1))
        paralist = cell.find("PARALIST")
        texts = []
        for p in (paralist.findall("P") if paralist is not None else []):
            texts.extend(b.strip() for b in _hwpml_paragraph(p) if b.strip())
        cells[key] = texts
    return grid_to_blocks(cells, spans)


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
            blocks.extend(grid_to_blocks(table["cells"], table["spans"]))
        table = None

    for tag, level, payload in iter_hwp_records(data):
        if table is not None and level <= table["level"]:
            flush_table()
        if tag == HWPTAG_CTRL_HEADER and payload[:4] == b" lbt" and table is None:
            table = {"level": level, "cells": {}, "spans": {}, "cell_level": level + 1, "current": None}
        elif tag == HWPTAG_LIST_HEADER and table is not None and level == table["cell_level"] and len(payload) >= 16:
            col, row, col_span = struct.unpack_from("<HHH", payload, 8)
            table["current"] = (row, col)
            table["cells"].setdefault((row, col), [])
            table["spans"][(row, col)] = col_span
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
