import unittest

from eval.golden import GoldenItem
from eval.retrieval import coverage, dedupe_refs, evaluate, first_hit_rank, summarize


class MetricTest(unittest.TestCase):
    def test_first_hit_rank_normalizes_refs(self):
        self.assertEqual(first_hit_rank(["여비규정 제1조", "여비규정 별표1"], ["여비규정 별표 1"]), 2)
        self.assertIsNone(first_hit_rank(["여비규정 제1조"], ["여비규정 제2조"]))

    def test_coverage_and_dedupe(self):
        refs = dedupe_refs(["A 제1조", "A 제1조", "A 제2조", "A 제3조"])
        self.assertEqual(refs, ["A 제1조", "A 제2조", "A 제3조"])
        self.assertEqual(coverage(refs, ["A 제1조", "A 제9조"], 5), 0.5)

    def test_summarize(self):
        s = summarize([{"rank": 1, "coverage@5": 1.0}, {"rank": 4, "coverage@5": 1.0}, {"rank": None, "coverage@5": 0.0}])
        self.assertAlmostEqual(s["recall@1"], 1 / 3)
        self.assertAlmostEqual(s["recall@5"], 2 / 3)
        self.assertAlmostEqual(s["mrr"], (1 + 0.25) / 3)

    def test_evaluate_ranks_by_article_not_chunk(self):
        items = [GoldenItem("q1", "fact", "여비", "질문", "답", ["A 제2조"])]
        hits = [(0.9, "A 제1조", {}), (0.8, "A 제1조", {}), (0.7, "A 제2조", {})]
        result = evaluate(items, lambda q, k: hits, k=5)[0]
        self.assertEqual(result["rank"], 2)  # 제1조 조각 2개는 한 순위로 묶임
        self.assertEqual(result["scores"], [0.9, 0.7])


if __name__ == "__main__":
    unittest.main()
