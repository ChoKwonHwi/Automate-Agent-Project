import json
import unittest
from dataclasses import dataclass

from rag.answer import answer_question, misused_evidence, select_blocks
from rag.llm import LLMResponse
from retrieval.search import Hit, Searcher


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def score(self, query, texts):
        return [self.scores.get(t, 0.0) for t in texts]


def hit(ref, text, score=0.6, rerank=None, part=0):
    return Hit(score, {"chunk_id": ref + str(part), "ref": ref, "org": "기관", "effective_date": "2024-01-01",
                       "text": text, "embed_text": text, "part": part, "doc_title": ref.split()[0]}, rerank=rerank)


class RerankStageTest(unittest.TestCase):
    def test_reorders_pool_by_relevance(self):
        stub = type("S", (), {})()
        stub.reranker = FakeReranker({"b": 0.9, "a": 0.1, "c": 0.5})
        hits = [hit("A 제1조", "a"), hit("A 제2조", "b"), hit("A 제3조", "c")]
        out = Searcher._rerank(stub, "질문", hits, k=2)
        self.assertEqual([h.ref for h in out], ["A 제2조", "A 제3조"])
        self.assertEqual(out[0].rerank, 0.9)


class MisusedEvidenceTest(unittest.TestCase):
    BLOCKS = [{"ref": "취업규정 제9조", "rerank": 0.124}, {"ref": "여비규정 제20조", "rerank": 0.0109},
              {"ref": "보수규정 제2조", "rerank": None}]

    def test_flags_quote_from_much_less_relevant_article(self):
        # W3 q010: 주소 변경신고 기한을 묻는데 이전비 신청 기한(여비규정 제20조)을 인용
        reason = misused_evidence([{"ref": "여비규정 제20조", "text": "…6개월 이내…"}], self.BLOCKS)
        self.assertIn("관련도가 낮음", reason)

    def test_passes_relevant_or_unscored_quotes(self):
        self.assertIsNone(misused_evidence([{"ref": "취업규정 제9조", "text": "…"}], self.BLOCKS))
        self.assertIsNone(misused_evidence([{"ref": "보수규정 제2조", "text": "…"}], self.BLOCKS))  # 정의 블록: 판단 보류
        no_scores = [{"ref": b["ref"], "rerank": None} for b in self.BLOCKS]
        self.assertIsNone(misused_evidence([{"ref": "여비규정 제20조", "text": "…"}], no_scores))

    def test_block_carries_max_rerank_of_parts(self):
        blocks = select_blocks([hit("A 제1조", "가" * 50, rerank=0.3, part=0), hit("A 제1조", "나" * 50, rerank=0.7, part=1)])
        self.assertEqual(blocks[0]["rerank"], 0.7)


class FakeLLM:
    def __init__(self, content):
        self.content = content

    def chat(self, messages, schema=None, think=False, temperature=0.1):
        return LLMResponse(self.content, 100, 50, 1000.0, 0.0)


class GateIntegrationTest(unittest.TestCase):
    def test_answer_quoting_irrelevant_article_is_refused(self):
        hits = [hit("취업규정 제9조", "직원은 거주지가 변경된 때에는 지체 없이 신고하여야 한다.", rerank=0.124),
                hit("여비규정 제20조", "이전비는 이전한 날의 다음 날부터 6개월 이내에 신청하여야 한다.", rerank=0.0109)]
        content = json.dumps({"quotes": [{"ref": "여비규정 제20조", "text": "이전비는 이전한 날의 다음 날부터 6개월 이내에 신청하여야 한다."}],
                              "answerable": True, "answer": "6개월 이내에 신고해야 합니다 (여비규정 제20조)",
                              "citations": ["여비규정 제20조"]})
        r = answer_question("이사하면 주소 변경신고는 며칠 안에 해?", lambda q, k: hits, FakeLLM(content))
        self.assertEqual((r.answerable, r.refused_by), (False, "verification"))
        self.assertTrue(any("관련도가 낮음" in w for w in r.warnings))


if __name__ == "__main__":
    unittest.main()
