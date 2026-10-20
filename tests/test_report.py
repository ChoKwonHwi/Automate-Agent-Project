import unittest
from collections import Counter

from ingestion.report import DocResult, review_sample
from ingestion.sources import Source


def make_result(doc: str, n_article: int, n_annex: int, n_supp: int) -> DocResult:
    src = Source(org="기관", org_code="o", title=doc, category="기타", effective_date="2026-01-01",
                 source_url="http://x", collected_at="2026-01-01", file_path=f"data/raw/o/{doc}.hwp", doc_id=doc)
    chunks = []
    for section, n in (("본칙", n_article), ("부칙", n_supp), ("별표", n_annex)):
        for i in range(n):
            chunks.append({"chunk_id": f"{doc}#{len(chunks)}", "section": section, "text": f"{section}{i} 첫줄\n끝줄{i}",
                           "doc_title": doc, "article_label": f"제{i + 1}조", "article_title": "", "part": 0,
                           "total_parts": 1})
    return DocResult(src, pages=1, chunks=chunks)


class ReviewSampleTest(unittest.TestCase):
    def test_stratified_and_spread_across_documents(self):
        results = [make_result("큰문서", 300, 200, 50), make_result("작은문서", 10, 2, 3), make_result("중간", 40, 0, 5)]
        sample = review_sample(results, size=30)
        self.assertEqual(len(sample), 30)
        by_section = Counter(s["chunk"]["section"] for s in sample)
        self.assertEqual(by_section, {"본칙": 21, "별표": 6, "부칙": 3})
        by_doc = Counter(s["chunk"]["doc_title"] for s in sample if s["chunk"]["section"] == "본칙")
        self.assertEqual(by_doc, {"큰문서": 7, "작은문서": 7, "중간": 7})

    def test_neighbor_context(self):
        sample = review_sample([make_result("문서", 3, 0, 0)], size=10)
        middle = next(s for s in sample if s["chunk"]["chunk_id"] == "문서#1")
        self.assertEqual((middle["prev"], middle["next"]), ("끝줄0", "본칙2 첫줄"))


if __name__ == "__main__":
    unittest.main()
