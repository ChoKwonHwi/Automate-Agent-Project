import tempfile
import unittest
from pathlib import Path

from retrieval.bm25 import BM25Index, tokenize
from retrieval.search import Hit, rrf_merge


def chunk(cid, text):
    return {"chunk_id": cid, "embed_text": text, "doc_title": "규정", "section": "본칙", "org": "기관"}


CHUNKS = [
    chunk("a", "[기관] 교육훈련규정 > 별표 4\n사내강사 일반강사 강사료 20,000 초과시간당 10,000원 추가"),
    chunk("b", "[기관] 보수규정 > 제2조(정의)\n연봉월액이란 연봉을 12로 나눈 금액을 말한다"),
    chunk("c", "[기관] 여비규정 > 제16조\n숙박비는 숙박하는 밤의 수에 따라 지급한다"),
]


class TokenizeTest(unittest.TestCase):
    def test_particles_removed_and_compounds_added(self):
        toks = tokenize("사내 일반강사가 받는 강사료는?")
        self.assertIn("강사료", toks)      # "강사"+"료" 복합 명사
        self.assertIn("강사", toks)
        self.assertNotIn("가", toks)       # 조사 제거
        self.assertNotIn("는", toks)


class BM25Test(unittest.TestCase):
    def test_exact_term_ranks_first_and_filter(self):
        index = BM25Index.build(CHUNKS)
        self.assertEqual(index.search("연봉월액이 얼마야?", 3)[0][0], 1)
        self.assertEqual(index.search("강사료 기준", 3)[0][0], 0)
        self.assertEqual(index.search("강사료", 3, keep=lambda c: c["chunk_id"] != "a"), [])

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bm25.pkl"
            first = BM25Index.build(CHUNKS, path)
            second = BM25Index.build(CHUNKS, path)  # 캐시에서 로드
            self.assertEqual(first.search("숙박비", 1), second.search("숙박비", 1))
            changed = BM25Index.build(CHUNKS[:2], path)  # 내용이 바뀌면 다시 구축
            self.assertEqual(len(changed.chunks), 2)


class RRFTest(unittest.TestCase):
    def test_merge_rewards_agreement(self):
        h = lambda cid, score, src: Hit(score, {"chunk_id": cid, "ref": cid}, sources=src)
        dense = [h("x", 0.7, "dense"), h("y", 0.6, "dense"), h("z", 0.5, "dense")]
        bm25 = [h("z", 0.5, "bm25"), h("w", 0.4, "bm25")]
        merged = rrf_merge(dense, bm25)
        self.assertEqual(merged[0].chunk["chunk_id"], "z")          # 양쪽에 다 나온 결과가 1위
        self.assertEqual(merged[0].sources, "dense+bm25")
        self.assertEqual(merged[0].score, 0.5)                      # score 는 dense 코사인 유지
        self.assertEqual({m.chunk["chunk_id"] for m in merged}, {"x", "y", "z", "w"})


if __name__ == "__main__":
    unittest.main()
