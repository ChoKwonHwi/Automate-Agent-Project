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
    def __init__(self, *contents):
        self.contents = list(contents)
        self.calls = []

    def chat(self, messages, schema=None, think=False, temperature=0.1):
        self.calls.append(messages[-1]["content"])
        return LLMResponse(self.contents.pop(0), 100, 50, 1000.0, 0.0)


class GateIntegrationTest(unittest.TestCase):
    def test_answer_quoting_irrelevant_article_is_retried_then_refused(self):
        # W3 q010: 주소 변경신고 기한을 묻는데 이전비 신청 기한(여비규정 제20조)을 인용 → 관련 근거만으로 재생성 → 거절
        hits = [hit("취업규정 제9조", "직원은 거주지가 변경된 때에는 지체 없이 신고하여야 한다.", rerank=0.124),
                hit("여비규정 제20조", "이전비는 이전한 날의 다음 날부터 6개월 이내에 신청하여야 한다.", rerank=0.0109)]
        misused = json.dumps({"quotes": [{"ref": "여비규정 제20조", "text": "이전비는 이전한 날의 다음 날부터 6개월 이내에 신청하여야 한다."}],
                              "answerable": True, "answer": "6개월 이내에 신고해야 합니다 (여비규정 제20조)",
                              "citations": ["여비규정 제20조"]})
        refuse = json.dumps({"quotes": [{"ref": "취업규정 제9조", "text": "직원은 거주지가 변경된 때에는 지체 없이 신고하여야 한다."}],
                             "answerable": False, "answer": "규정에서 확인되지 않는다. 지체 없이 신고해야 한다는 규정만 있다.",
                             "citations": []})
        r = answer_question("이사하면 주소 변경신고는 며칠 안에 해?", lambda q, k: hits, FakeLLM(misused, refuse))
        self.assertEqual((r.answerable, r.refused_by, r.retried), (False, "llm", True))


class RetryTest(unittest.TestCase):
    HITS = [hit("계약사무세칙 제3조", "발주부서란 계약요청을 하는 부서를 말한다.", rerank=0.999),
            hit("위임전결세칙 별표2", "- 15. 계약관리 > 계약체결 → 전결권자·부서장", rerank=0.118)]

    def test_misused_evidence_triggers_one_retry_without_low_relevance_blocks(self):
        bad = json.dumps({"quotes": [{"ref": "위임전결세칙 별표2", "text": "- 15. 계약관리 > 계약체결 → 전결권자·부서장"}],
                          "answerable": True, "answer": "발주부서입니다", "citations": ["위임전결세칙 별표2"]})
        good = json.dumps({"quotes": [{"ref": "계약사무세칙 제3조", "text": "발주부서란 계약요청을 하는 부서를 말한다."}],
                           "answerable": True, "answer": "발주부서입니다", "citations": ["계약사무세칙 제3조"]})
        llm = FakeLLM(bad, good)
        r = answer_question("계약요청하는 부서는?", lambda q, k: self.HITS, llm)
        self.assertTrue(r.answerable and r.retried)
        self.assertEqual(r.citations, ["계약사무세칙 제3조"])
        self.assertEqual(len(llm.calls), 2)
        self.assertNotIn("위임전결세칙 별표2", llm.calls[1])  # 재생성 때는 관련도 낮은 근거를 뺐다
        self.assertIn("재생성", r.warnings[0])

    def test_no_retry_for_other_verification_failures(self):
        content = json.dumps({"quotes": [], "answerable": True, "answer": "답", "citations": []})
        llm = FakeLLM(content)
        r = answer_question("질문", lambda q, k: self.HITS, llm)
        self.assertEqual((r.refused_by, r.retried, len(llm.calls)), ("verification", False, 1))


if __name__ == "__main__":
    unittest.main()
