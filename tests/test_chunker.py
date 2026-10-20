import unittest

from ingestion.chunker import chunk_document, parse_annex_heading, parse_article_heading, split_blocks
from ingestion.clean import clean_pages
from ingestion.loaders import Page

META = {"doc_id": "test-여비규정", "org": "테스트기관", "title": "여비규정", "category": "여비",
        "effective_date": "2025-01-01", "source_url": "https://example.org"}


def lines_of(text: str, page: int = 1):
    return [(l.strip(), page) for l in text.strip().splitlines() if l.strip()]


class ArticleHeadingTest(unittest.TestCase):
    def test_heading_with_title(self):
        h = parse_article_heading("제12조(숙박비) ① 숙박비는 별표 2에 따른다.")
        self.assertEqual((h.number, h.sub, h.title), (12, 0, "숙박비"))

    def test_branch_article(self):
        h = parse_article_heading("제12조의2(숙박비 특례)")
        self.assertEqual((h.label, h.title), ("제12조의2", "숙박비 특례"))

    def test_deleted_article(self):
        h = parse_article_heading("제5조 삭제 <2020.1.1>")
        self.assertEqual((h.label, h.title, h.rest), ("제5조", "", "삭제 <2020.1.1>"))

    def test_cross_references_are_not_headings(self):
        for line in ["제12조에 따른 숙박비는", "제12조제1항의 규정에도", "제12조 제2항에 따라",
                     "제12조 및 제13조의 규정은", "제12조의 규정에 의하여", "제3조, 제4조를 준용한다."]:
            self.assertIsNone(parse_article_heading(line), line)


class AnnexHeadingTest(unittest.TestCase):
    def test_annex_forms(self):
        self.assertEqual(parse_annex_heading("[별표 1] 국내여비 지급표(제12조 관련)"), "별표 1")
        self.assertEqual(parse_annex_heading("【별표 2의2】"), "별표 2의2")
        self.assertEqual(parse_annex_heading("[별지 제1호 서식]"), "별지 제1호")
        self.assertEqual(parse_annex_heading("[별지 제3호서식]"), "별지 제3호")
        self.assertEqual(parse_annex_heading("별표 3 <개정 2024.1.1>"), "별표 3")

    def test_annex_reference_is_not_heading(self):
        self.assertIsNone(parse_annex_heading("별표 1에 따른 금액을 지급한다."))


SAMPLE = """
여비규정
제정 2010.1.1
제1장 총칙
제1조(목적) 이 규정은 임직원의 여비 지급에 관한 사항을 정함을 목적으로 한다.
제2조(정의) 이 규정에서 사용하는 용어의 뜻은 다음과 같다.
1. "출장"이란 업무를 위하여 근무지를 떠나는 것을 말한다.
제2장 국내여비
제3조(숙박비) ① 숙박비는 별표 1에 따라 지급한다.
제2조에 따른 출장의 경우에도 같다.
② 숙박하지 아니한 경우에는 지급하지 아니한다.
제3조의2(숙박비 특례) 사장이 인정하는 경우 상한을 초과하여 지급할 수 있다.
제4조 삭제 <2020.1.1>
부칙 <2010.1.1>
이 규정은 2010년 1월 1일부터 시행한다.
부칙 <2020.1.1>
제1조(시행일) 이 규정은 2020년 1월 1일부터 시행한다.
제2조(경과조치) 종전 규정에 따른다.
[별표 1] 숙박비 지급표(제3조 관련)
| 구분 | 서울 | 광역시 |
|---|---|---|
| 임원 | 100,000 | 80,000 |
"""


class SplitBlocksTest(unittest.TestCase):
    def setUp(self):
        self.blocks = split_blocks(lines_of(SAMPLE))

    def test_block_labels(self):
        labels = [(b.section, b.label) for b in self.blocks]
        self.assertEqual(labels, [
            ("전문", "전문"),
            ("본칙", "제1조"), ("본칙", "제2조"), ("본칙", "제3조"), ("본칙", "제3조의2"), ("본칙", "제4조"),
            ("부칙", "부칙"),
            ("부칙", "제1조"), ("부칙", "제2조"),
            ("별표", "별표 1"),
        ])

    def test_cross_reference_line_stays_in_article(self):
        art3 = next(b for b in self.blocks if b.label == "제3조")
        self.assertIn("제2조에 따른 출장의 경우에도 같다.", [l for l, _ in art3.lines])

    def test_chapter_metadata_and_removed_from_text(self):
        art3 = next(b for b in self.blocks if b.label == "제3조")
        self.assertEqual(art3.chapter, "제2장 국내여비")
        art2 = next(b for b in self.blocks if b.label == "제2조" and b.section == "본칙")
        self.assertNotIn("제2장 국내여비", [l for l, _ in art2.lines])


class ChunkDocumentTest(unittest.TestCase):
    def test_metadata_and_ids(self):
        chunks = chunk_document(lines_of(SAMPLE), META)
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        art3 = next(c for c in chunks if c["article_label"] == "제3조")
        self.assertEqual(art3["article_title"], "숙박비")
        self.assertTrue(art3["embed_text"].startswith("[테스트기관] 여비규정 > 제2장 국내여비 > 제3조(숙박비)"))
        supp = [c["article_label"] for c in chunks if c["section"] == "부칙"]
        self.assertEqual(supp, ["부칙1", "부칙2 제1조", "부칙2 제2조"])

    def test_long_article_split_by_paragraph(self):
        body = "\n".join(f"{m} " + "가" * 300 for m in "①②③④")
        chunks = chunk_document(lines_of("제7조(일비)\n" + body), META, max_chars=700)
        art = [c for c in chunks if c["article_label"] == "제7조"]
        self.assertEqual(len(art), 2)
        self.assertTrue(art[1]["text"].startswith("제7조(일비) (계속)\n③"))
        self.assertTrue(all(c["total_parts"] == 2 for c in art))

    def test_oversized_paragraph_split_by_line(self):
        body = "① 첫째 항\n" + "\n".join("나" * 100 for _ in range(30)) + "\n② 둘째 항"
        chunks = chunk_document(lines_of("제8조(식비)\n" + body), META, max_chars=500)
        self.assertTrue(all(c["char_len"] <= 500 + len("제8조(식비) (계속)\n") for c in chunks))
        self.assertGreaterEqual(len(chunks), 6)

    def test_page_tracking(self):
        lines = lines_of("제1조(목적) 목적이다.", 1) + lines_of("계속되는 내용", 2) + lines_of("제2조(정의) 정의다.", 2)
        chunks = chunk_document(lines, META)
        first = next(c for c in chunks if c["article_label"] == "제1조")
        self.assertEqual((first["page_start"], first["page_end"]), (1, 2))


class CleanPagesTest(unittest.TestCase):
    def test_removes_page_numbers_and_repeated_headers(self):
        pages = [Page(i, f"여비규정\n제{i}조(항목{i}) 내용\n- {i} -") for i in range(1, 5)]
        lines = [l for l, _ in clean_pages(pages)]
        self.assertEqual(lines, [f"제{i}조(항목{i}) 내용" for i in range(1, 5)])


    def test_law_go_kr_two_page_document(self):
        title = "○○공단 여비규정"
        pages = [
            Page(1, f"{title}\n{title}\n[시행 2018. 7. 1.]\n제1조(목적) 목적이다.\n부칙 <제1호,2010.7.29.>\n"
                    "이 규정은 시행한다.\n법제처 1 국가법령정보센터"),
            Page(2, f"{title}\n부칙 <제2호,2014.9.18.>\n이 규정은 시행한다.\n법제처 2 국가법령정보센터"),
        ]
        lines = [l for l, _ in clean_pages(pages)]
        self.assertNotIn(title, lines)
        self.assertFalse(any("국가법령정보센터" in l for l in lines))
        self.assertIn("부칙 <제2호,2014.9.18.>", lines)
        self.assertIn("부칙 <제1호,2010.7.29.>", lines)


if __name__ == "__main__":
    unittest.main()
