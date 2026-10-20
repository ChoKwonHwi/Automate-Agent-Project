import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from ingestion.loaders import decode_hwp_para_text, hwp_section_text, load_document, table_to_markdown

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HS = "http://www.hancom.co.kr/hwpml/2011/section"


def hwp_record(tag: int, payload: bytes, level: int = 0) -> bytes:
    return struct.pack("<I", tag | (level << 10) | (len(payload) << 20)) + payload


def hwp_para(text: str, level: int) -> bytes:
    return hwp_record(66, b"\x00" * 22, level) + hwp_record(67, (text + "\r").encode("utf-16-le"), level + 1)


def hwp_cell(col: int, row: int, col_span: int, text: str) -> bytes:
    header = struct.pack("<HHI", 1, 0, 0x20) + struct.pack("<HHHH", col, row, col_span, 1) + b"\x00" * 31
    return hwp_record(72, header, 2) + hwp_para(text, 2)


class HwpBinaryTest(unittest.TestCase):
    def test_decode_skips_wide_controls(self):
        # 표 컨트롤(11) 은 8 wchar 를 차지하므로 통째로 건너뛴다
        units = [ord(c) for c in "제1조"] + [11, 0x74, 0x62, 0x6C, 0x20, 0, 0, 11] + [ord(c) for c in "(목적)"] + [13]
        payload = struct.pack(f"<{len(units)}H", *units)
        self.assertEqual(decode_hwp_para_text(payload), "제1조(목적)")

    def test_section_text_reads_para_text_records(self):
        para = lambda s: hwp_record(67, (s + "\r").encode("utf-16-le"))
        data = hwp_record(66, b"\x00" * 22) + para("제1조(목적)") + para("이 규정은 …")
        self.assertEqual(hwp_section_text(data), "제1조(목적)\n이 규정은 …")

    def test_table_rebuilt_as_markdown_with_full_width_note_outside(self):
        data = (
            hwp_para("[별표 3] 이전비 정액표", 0)
            + hwp_record(66, b"\x00" * 22, 0)
            + hwp_record(71, b" lbt" + b"\x00" * 40, 1)
            + hwp_record(77, struct.pack("<IHH", 0, 3, 2), 2)
            + hwp_cell(0, 0, 1, "구분") + hwp_cell(1, 0, 1, "100㎞ 미만")
            + hwp_cell(0, 1, 1, "임원") + hwp_cell(1, 1, 1, "300")
            + hwp_cell(0, 2, 2, "비고 : 선편 ㎞당 1,500원 추가")
            + hwp_para("끝", 0)
        )
        self.assertEqual(hwp_section_text(data).splitlines(), [
            "[별표 3] 이전비 정액표",
            "| 구분 | 100㎞ 미만 |", "|---|---|", "| 임원 | 300 |",
            "비고 : 선편 ㎞당 1,500원 추가",
            "끝",
        ])


class HwpxTest(unittest.TestCase):
    def test_paragraphs_and_table(self):
        xml = f"""<hs:sec xmlns:hs="{HS}" xmlns:hp="{HP}">
          <hp:p><hp:run><hp:t>제1조(목적) 목적이다.</hp:t></hp:run></hp:p>
          <hp:p><hp:run><hp:tbl>
            <hp:tr><hp:tc><hp:subList><hp:p><hp:run><hp:t>구분</hp:t></hp:run></hp:p></hp:subList></hp:tc>
                   <hp:tc><hp:subList><hp:p><hp:run><hp:t>금액</hp:t></hp:run></hp:p></hp:subList></hp:tc></hp:tr>
            <hp:tr><hp:tc><hp:subList><hp:p><hp:run><hp:t>임원</hp:t></hp:run></hp:p></hp:subList></hp:tc>
                   <hp:tc><hp:subList><hp:p><hp:run><hp:t>100,000</hp:t></hp:run></hp:p></hp:subList></hp:tc></hp:tr>
          </hp:tbl></hp:run></hp:p>
        </hs:sec>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "규정.hwpx"
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("Contents/section0.xml", xml)
            pages = load_document(path)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0].text.splitlines(), [
            "제1조(목적) 목적이다.", "| 구분 | 금액 |", "|---|---|", "| 임원 | 100,000 |",
        ])


class MarkdownTableTest(unittest.TestCase):
    def test_pads_rows_and_escapes_pipes(self):
        md = table_to_markdown([["a", "b|c"], ["1"], [None, ""]])
        self.assertEqual(md, "| a | b/c |\n|---|---|\n| 1 |  |")


if __name__ == "__main__":
    unittest.main()
