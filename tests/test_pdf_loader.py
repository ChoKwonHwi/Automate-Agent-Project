import tempfile
import unittest
from pathlib import Path

from ingestion.loaders import load_document

try:
    import pymupdf
except ImportError:  # 가상환경 밖(예: 시스템 Python)에서는 건너뛴다
    pymupdf = None


def make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), "제12조(숙박비) 숙박비는 별표 1에 따른다.", fontname="korea", fontsize=11)
    # 3x2 격자 표
    x = [50, 200, 350]
    y = [100, 130, 160, 190]
    for xi in x:
        page.draw_line((xi, y[0]), (xi, y[-1]))
    for yi in y:
        page.draw_line((x[0], yi), (x[-1], yi))
    cells = [["구분", "금액"], ["임원", "100,000"], ["직원", "70,000"]]
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            page.insert_text((x[c] + 5, y[r] + 20), text, fontname="korea", fontsize=11)
    page.insert_text((50, 230), "제13조(식비) 식비는 1일 25,000원으로 한다.", fontname="korea", fontsize=11)
    doc.save(path)
    doc.close()


@unittest.skipIf(pymupdf is None, "PyMuPDF 미설치")
class PdfLoaderTest(unittest.TestCase):
    def test_text_and_table_in_reading_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "규정.pdf"
            make_pdf(path)
            pages = load_document(path)
        lines = [l.strip() for l in pages[0].text.splitlines() if l.strip()]
        self.assertEqual(lines[0], "제12조(숙박비) 숙박비는 별표 1에 따른다.")
        self.assertIn("| 구분 | 금액 |", lines)
        self.assertIn("| 임원 | 100,000 |", lines)
        self.assertEqual(lines[-1], "제13조(식비) 식비는 1일 25,000원으로 한다.")
        # 표 셀 텍스트가 일반 텍스트로 중복되지 않아야 한다
        self.assertFalse(any(l == "임원" or l.startswith("임원 ") for l in lines))


if __name__ == "__main__":
    unittest.main()
