import json
import unittest
from dataclasses import dataclass

from rag.answer import REFUSAL_TEXT, answer_question, canonical_citation, keyword_overlap, select_blocks, validate_citations
from rag.llm import LLMResponse
from rag.prompts import build_messages


@dataclass
class FakeHit:
    score: float
    chunk: dict

    @property
    def ref(self):
        return self.chunk["ref"]


def hit(score, ref, text, part=0):
    return FakeHit(score, {"ref": ref, "org": "기관", "effective_date": "2024-02-02", "text": text, "part": part})


class FakeLLM:
    def __init__(self, *contents):
        self.contents = list(contents)
        self.calls = []

    def chat(self, messages, schema=None, temperature=0.1):
        self.calls.append(messages)
        return LLMResponse(self.contents.pop(0), 1000, 50, 1500.0, 0.0)


HITS = [hit(0.7, "여비규정 별표1", "숙박비 서울 100,000"), hit(0.65, "여비규정 제16조", "①일비…", 0),
        hit(0.6, "여비규정 제16조", "제16조 (계속)\n③공단차량 2분의 1", 1), hit(0.5, "인사규정 제1조", "목적")]


class SelectBlocksTest(unittest.TestCase):
    def test_merges_parts_and_keeps_rank_order(self):
        blocks = select_blocks(HITS)
        self.assertEqual([b["ref"] for b in blocks], ["여비규정 별표1", "여비규정 제16조", "인사규정 제1조"])
        self.assertIn("③공단차량", blocks[1]["text"])

    def test_char_budget_truncates(self):
        blocks = select_blocks([hit(0.9, "A 제1조", "가" * 3000), hit(0.8, "A 제2조", "나" * 3000)],
                               char_budget=4000, block_cap=3000)
        self.assertEqual(len(blocks[0]["text"]), 3000)
        self.assertTrue(blocks[1]["text"].endswith("…(이하 생략)"))

    def test_block_cap_keeps_best_parts_in_order(self):
        # 긴 조문의 조각 5개(검색 순위: 0, 3, 1, 4, 2)가 상한 때문에 앞의 3개만 남고 원래 순서로 이어진다
        long = [hit(0.9 - i / 100, "A 제57조", f"조각{p}" + "가" * 500, p) for i, p in enumerate([0, 3, 1, 4, 2])]
        blocks = select_blocks(long + [hit(0.5, "A 제58조", "다음 조")], block_cap=1600)
        text = blocks[0]["text"]
        self.assertLess(text.index("조각0"), text.index("조각1"))
        self.assertLess(text.index("조각1"), text.index("…(중략)"))
        self.assertLess(text.index("…(중략)"), text.index("조각3"))
        self.assertNotIn("조각4", text)
        self.assertEqual([b["ref"] for b in blocks], ["A 제57조", "A 제58조"])  # 긴 조문이 예산을 독차지하지 않음

    def test_keyword_aware_part_selection(self):
        parts = [hit(0.70, "B 별표4", "사외강사 상한 400,000 " + "가" * 1100, 0),
                 hit(0.68, "B 별표4", "사내강사 일반강사 20,000 초과시간당 10,000 " + "나" * 1150, 1),
                 hit(0.66, "B 별표4", "원고료", 2)]
        q = "사내 일반강사가 3시간 강의하면 강사료는?"
        self.assertGreater(keyword_overlap(q, parts[1].chunk["text"]), keyword_overlap(q, parts[0].chunk["text"]))
        text = select_blocks(parts, block_cap=1800, question=q)[0]["text"]
        self.assertIn("일반강사", text)
        self.assertNotIn("사외강사", select_blocks(parts, block_cap=1300, question=q)[0]["text"])

    def test_prompt_contains_refs(self):
        user = build_messages("질문", select_blocks(HITS))[1]["content"]
        self.assertIn("[근거 1] 여비규정 별표1 (기관, 시행 2024-02-02)", user)


class AnswerTest(unittest.TestCase):
    def test_low_score_refuses_without_llm(self):
        llm = FakeLLM()
        r = answer_question("주식 추천", lambda q, k: [hit(0.3, "A 제1조", "x")], llm)
        self.assertEqual((r.answerable, r.refused_by, r.answer), (False, "retrieval", REFUSAL_TEXT))
        self.assertEqual(llm.calls, [])

    def test_answer_with_valid_and_invented_citations(self):
        content = json.dumps({"quotes": [{"ref": "여비규정 별표1", "text": "숙박비 서울 100,000"}],
                              "answerable": True, "answer": "10만원 (여비규정 별표1)",
                              "citations": ["여비규정 별표 1", "여비규정 제99조"]})
        r = answer_question("숙박비?", lambda q, k: HITS, FakeLLM(content))
        self.assertTrue(r.answerable)
        self.assertEqual(r.citations, ["여비규정 별표1"])
        self.assertEqual(r.invalid_citations, ["여비규정 제99조"])
        self.assertTrue(any("근거에 없는 조항" in w for w in r.warnings))

    def test_llm_refusal(self):
        content = json.dumps({"answerable": False, "answer": "규정에 없음", "citations": ["여비규정 제16조"]})
        r = answer_question("재택근무?", lambda q, k: HITS, FakeLLM(content))
        self.assertEqual((r.answerable, r.refused_by, r.citations), (False, "llm", []))

    def test_retry_once_on_bad_json(self):
        good = json.dumps({"quotes": [{"ref": "여비규정 제16조", "text": "③공단차량 2분의 1"}],
                           "answerable": True, "answer": "답", "citations": ["여비규정 제16조"]})
        llm = FakeLLM("not json", good)
        r = answer_question("질문", lambda q, k: HITS, llm)
        self.assertEqual((r.answerable, len(llm.calls)), (True, 2))

    def test_canonical_citation_variants(self):
        blocks = [{"ref": "회계규정 제15조"}, {"ref": "교육훈련규정 별표4"}, {"ref": "보수규정 제2조"}]
        cases = {
            "회계규정 제15조 ②": "회계규정 제15조",
            "보수규정 제2조 제6항 (연봉월액 정의)": "보수규정 제2조",
            "[근거 2] 별표 4 제60조제2항": "교육훈련규정 별표4",
            "근거 3": "보수규정 제2조",
            "제15조": "회계규정 제15조",
            "별표 4": "교육훈련규정 별표4",
        }
        for cited, expected in cases.items():
            self.assertEqual(canonical_citation(cited, blocks), expected, cited)
        for cited in ("여비규정 제16조", "근거 9", "제99조"):
            self.assertIsNone(canonical_citation(cited, blocks), cited)

    def test_ungrounded_number_warning(self):
        content = json.dumps({"quotes": [{"ref": "여비규정 별표1", "text": "숙박비 서울 100,000"}],
                              "answerable": True, "answer": "숙박비는 100,000원, 합계 230,000원 (여비규정 별표1)",
                              "citations": ["여비규정 별표1"]})
        r = answer_question("숙박비는?", lambda q, k: HITS, FakeLLM(content))
        self.assertTrue(r.answerable)  # 금액 계산값은 거절하지 않고 경고만 (W4 계산 도구)
        self.assertEqual(r.ungrounded_numbers, [230000])

    def test_invented_deadline_is_refused_by_verification(self):
        # W2 q003: 근거 문장은 맞게 인용하면서 원문에 없는 "8근무시간"을 지어낸 경우
        content = json.dumps({"quotes": [{"ref": "여비규정 별표1", "text": "숙박비 서울 100,000"}],
                              "answerable": True, "answer": "본사는 8근무시간 이내에 지시해야 한다 (여비규정 별표1)",
                              "citations": ["여비규정 별표1"]})
        r = answer_question("본사는 며칠 안에 지시해?", lambda q, k: HITS, FakeLLM(content))
        self.assertEqual((r.answerable, r.refused_by), (False, "verification"))
        self.assertIn("8근무시간", " ".join(r.warnings))
        self.assertIn("8근무시간", r.raw_answer)

    def test_no_verified_quote_is_refused(self):
        content = json.dumps({"quotes": [{"ref": "여비규정 별표1", "text": "숙박비는 지역 구분 없이 50만원으로 한다"}],
                              "answerable": True, "answer": "50만원입니다", "citations": ["여비규정 별표1"]})
        r = answer_question("숙박비?", lambda q, k: HITS, FakeLLM(content))
        self.assertEqual((r.answerable, r.refused_by, len(r.rejected_quotes)), (False, "verification", 1))

    def test_validate_citations_dedupes(self):
        blocks = [{"ref": "여비규정 별표1"}]
        self.assertEqual(validate_citations(["여비규정 별표1", "여비규정 별표 1"], blocks), (["여비규정 별표1"], []))


if __name__ == "__main__":
    unittest.main()
