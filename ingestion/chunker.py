"""정제된 규정 텍스트를 조(條) 단위 청크로 나눈다.

구조 인식
- 장 / 절: 청크 메타데이터(chapter)로만 사용
- 조: "제12조(숙박비)", "제12조의2(...)", "제5조 삭제" 를 청크 경계로 사용
- 부칙: 조 번호가 다시 1부터 시작하므로 별도 섹션으로 관리
- 별표 / 별지: "[별표 1]" 단위로 청크

본문 중 "제12조에 따라" 처럼 줄 첫머리에 온 조문 인용은 경계로 보지 않는다
(조사가 붙어 있거나 조 번호가 앞 조보다 작으면 인용으로 판단).
긴 조문은 항(①②…) 단위로 묶어 max_chars 이하로 나눈다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .clean import Line

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

_ARTICLE_HEAD_RE = re.compile(r"^제\s*(\d+)\s*조(?:\s*의\s*(\d+))?")
_TITLE_RE = re.compile(r"^\s*[(（]([^)）\n]{1,60})[)）]")
_CHAPTER_RE = re.compile(r"^제\s*(\d+)\s*(장|절)(?:\s+(.{0,40}))?$")
_SUPPLEMENT_RE = re.compile(r"^부\s*칙\s*(?:[<(（\[〈].*)?$")
_ANNEX_BRACKET_RE = re.compile(
    r"^[\[【<〔〈]\s*(별\s*표|별\s*지)\s*(제?\s*\d+\s*호?(?:\s*의\s*\d+)?)?\s*(?:서\s*식)?\s*[\]】>〕〉]\s*(.*)$"
)
_ANNEX_INDEX_RE = re.compile(r"^별\s*표\s*/\s*서\s*식$")
_ANNEX_PLAIN_RE = re.compile(r"^(별\s*표|별\s*지)\s*(\d+(?:\s*의\s*\d+)?)(.*)$")

# 붙어 쓰이면 인용으로 보는 조사 / 띄어 쓰여도 인용으로 보는 연결어
_ATTACHED_PARTICLE_RE = re.compile(r"^(?:에|의|을|를|와|과|로|으로|이|가|은|는|부터|까지|에서|·|,|ㆍ|~|및)")
_SPACED_CONNECTOR_RE = re.compile(r"^\s+(?:및|또는|내지|부터|까지|ㆍ|·|,|~|에\s*따)")
_CROSS_REF_RE = re.compile(r"^\s*제\s*\d")


@dataclass
class ArticleHeading:
    number: int
    sub: int
    title: str
    rest: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.number, self.sub)

    @property
    def label(self) -> str:
        return f"제{self.number}조" + (f"의{self.sub}" if self.sub else "")


def parse_article_heading(line: str) -> ArticleHeading | None:
    m = _ARTICLE_HEAD_RE.match(line)
    if not m:
        return None
    rem = line[m.end():]
    if _CROSS_REF_RE.match(rem):  # "제5조제1항", "제5조 제2항에"
        return None
    title = ""
    tm = _TITLE_RE.match(rem)
    if tm:
        title = tm.group(1).strip()
        rem = rem[tm.end():]
    if rem and not rem[0].isspace() and _ATTACHED_PARTICLE_RE.match(rem):
        return None
    if _SPACED_CONNECTOR_RE.match(rem):
        return None
    return ArticleHeading(int(m.group(1)), int(m.group(2) or 0), title, rem.strip())


def parse_annex_heading(line: str) -> str | None:
    m = _ANNEX_BRACKET_RE.match(line)
    if not m:
        m = _ANNEX_PLAIN_RE.match(line)
        if not m:
            return None
        rem = m.group(3)
        if rem and not rem[0].isspace() and _ATTACHED_PARTICLE_RE.match(rem):
            return None
        if _SPACED_CONNECTOR_RE.match(rem):
            return None
    kind = re.sub(r"\s+", "", m.group(1))
    number = re.sub(r"\s+", "", m.group(2) or "")
    return f"{kind} {number}".strip()


@dataclass
class Block:
    section: str  # 전문 / 본칙 / 부칙 / 별표
    label: str
    title: str
    chapter: str
    supplement_no: int = 0
    lines: list[Line] = field(default_factory=list)
    has_heading: bool = True

    @property
    def body_lines(self) -> list[Line]:
        return self.lines[1:] if self.has_heading else self.lines


def split_blocks(lines: list[Line]) -> list[Block]:
    blocks: list[Block] = []
    current = Block("전문", "전문", "", "", has_heading=False)
    chapter, section_title = "", ""
    last_key: tuple[int, int] | None = None
    supplement_no = 0
    mode = "본칙"

    def close():
        if current.lines:
            blocks.append(current)

    for line, page in lines:
        if _ANNEX_INDEX_RE.match(line):  # 국가법령정보센터 본문 끝의 "별표 / 서식" 목차 제목
            continue
        if mode != "별표" and (name := parse_chapter(line)):
            if "장" in name.split()[0]:
                chapter, section_title = name, ""
            else:
                section_title = name
            continue

        if _SUPPLEMENT_RE.match(line):
            close()
            supplement_no += 1
            mode, last_key = "부칙", None
            current = Block("부칙", "부칙", line, "", supplement_no, [(line, page)])
            continue

        annex = parse_annex_heading(line)
        if annex:
            close()
            mode = "별표"
            current = Block("별표", annex, line, "", lines=[(line, page)])
            continue

        if mode != "별표":
            heading = parse_article_heading(line)
            if heading and (last_key is None or heading.key > last_key):
                close()
                last_key = heading.key
                if mode == "본칙":
                    chap = " > ".join(x for x in (chapter, section_title) if x)
                    current = Block("본칙", heading.label, heading.title, chap, lines=[(line, page)])
                else:
                    current = Block("부칙", heading.label, heading.title, "", supplement_no, [(line, page)])
                continue

        current.lines.append((line, page))

    close()
    # 헤딩만 있고 내용이 없는 블록은 버린다
    # - "부칙 <2020.1.1>" 뒤에 부칙 조문이 이어지는 경우
    # - 본문 끝 "별표 / 서식" 목차의 별표 제목 (실제 별표는 별도 파일로 수집)
    def heading_only(b: Block) -> bool:
        return not b.body_lines and (b.label == "부칙" or b.section == "별표")

    return [b for b in blocks if not heading_only(b)]


def parse_chapter(line: str) -> str | None:
    m = _CHAPTER_RE.match(line)
    if not m:
        return None
    name = m.group(3) or ""
    if _SPACED_CONNECTOR_RE.match(" " + name) or line.endswith((".", "다")):
        return None
    return f"제{m.group(1)}{m.group(2)} {name}".strip()


def _pack_lines(lines: list[Line], max_chars: int) -> list[list[Line]]:
    """항(①②…) 단위로 묶어 max_chars 이하가 되도록 나눈다. 항이 없으면 줄 단위."""
    units: list[list[Line]] = []
    for line in lines:
        if not units or (line[0][:1] in CIRCLED):
            units.append([line])
        else:
            units[-1].append(line)
    # 항 하나가 max_chars 를 넘으면 그 항은 줄 단위로 다시 나눈다
    units = [
        piece
        for unit in units
        for piece in ([unit] if sum(len(l[0]) + 1 for l in unit) <= max_chars else [[l] for l in unit])
    ]

    parts: list[list[Line]] = []
    buf: list[Line] = []
    size = 0
    for unit in units:
        unit_size = sum(len(l[0]) + 1 for l in unit)
        if buf and size + unit_size > max_chars:
            parts.append(buf)
            buf, size = [], 0
        buf.extend(unit)
        size += unit_size
    if buf:
        parts.append(buf)
    return parts


def _continuation_header(block: Block, start: int) -> str:
    """분할된 조각 앞에 붙일 머리. 조각이 표 중간에서 시작하면 그 표의 머리행을 다시 붙인다."""
    head = f"{block.lines[0][0]} (계속)" if block.has_heading else "(계속)"
    lines = [l for l, _ in block.lines]
    if start < len(lines) and lines[start].startswith("|"):
        for i in range(start - 1, 0, -1):
            if lines[i].startswith("|---") and lines[i - 1].startswith("|"):
                return "\n".join([head, lines[i - 1], lines[i]])
            if not lines[i].startswith("|"):
                break
    return head


def chunk_document(lines: list[Line], meta: dict, max_chars: int = 1200) -> list[dict]:
    doc_id = meta["doc_id"]
    chunks: list[dict] = []
    for idx, block in enumerate(split_blocks(lines)):
        text_len = sum(len(l) + 1 for l, _ in block.lines)
        parts = [block.lines] if text_len <= max_chars else _pack_lines(block.lines, max_chars)
        start = 0
        for part_no, part in enumerate(parts):
            text = "\n".join(l for l, _ in part)
            if part_no > 0:
                text = f"{_continuation_header(block, start)}\n{text}"
            start += len(part)
            label = block.label
            if block.section == "부칙":
                label = f"부칙{block.supplement_no} {block.label}" if block.label != "부칙" else f"부칙{block.supplement_no}"
            context = " > ".join(
                x for x in (f"[{meta.get('org', '')}] {meta.get('title', '')}".strip(), block.chapter,
                            f"{label}({block.title})" if block.title and block.section == "본칙" else label) if x
            )
            chunks.append({
                "chunk_id": f"{doc_id}#{idx:04d}-{part_no}",
                "doc_id": doc_id,
                "org": meta.get("org", ""),
                "doc_title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "effective_date": meta.get("effective_date", ""),
                "source_url": meta.get("source_url", ""),
                "section": block.section,
                "chapter": block.chapter,
                "article_label": label,
                "article_title": block.title if block.section == "본칙" else "",
                "part": part_no,
                "total_parts": len(parts),
                "page_start": part[0][1],
                "page_end": part[-1][1],
                "text": text,
                "embed_text": f"{context}\n{text}",
                "char_len": len(text),
            })
    return chunks
