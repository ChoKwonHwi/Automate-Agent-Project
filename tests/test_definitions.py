import unittest

from rag.answer import with_definitions
from rag.definitions import build_index, matching_definitions


def chunk(doc, label, title, text, section="본칙"):
    return {"doc_title": doc, "article_label": label, "article_title": title, "section": section, "text": text,
            "org": "기관", "effective_date": "2025-12-22"}


CHUNKS = [
    chunk("보수규정", "제2조", "정의", "제2조(정의) 이 규정에서 사용하는 용어의 뜻은 다음과 같다.\n"
          '1. "보수"란 기준급과 수당을 합산한 금액을 말한다.\n'
          '6. "연봉월액"이란 연봉을 12로 나눈 금액을 말한다.\n'
          '7. "연봉의 일할계산"이란 연봉월액을 그 달의 일수로 나누어 계산하는 것을 말한다'),
    chunk("민원처리예규", "제2조", "정의", '2. "민원인"이라 함은 공단에 민원을 제기하는 자를 말한다.'),
    chunk("보수규정", "제29조", "연봉", "제29조(연봉) 연봉은 …"),
]


class DefinitionsTest(unittest.TestCase):
    def setUp(self):
        self.index = build_index(CHUNKS)

    def test_index_skips_short_terms_and_handles_variants(self):
        terms = [d.term for d in self.index["보수규정"]]
        self.assertEqual(terms, ["연봉월액", "연봉의 일할계산"])  # "보수"(2글자)는 제외
        self.assertEqual([d.term for d in self.index["민원처리예규"]], ["민원인"])

    def test_matching_uses_variants_and_prefers_longer_terms(self):
        found = matching_definitions("연봉 6천만원이면 연봉월액이랑 연봉 일할계산액은?", ["보수규정"], self.index)
        self.assertEqual([d.term for d in found], ["연봉의 일할계산", "연봉월액"])
        self.assertEqual(matching_definitions("연봉월액?", ["민원처리예규"], self.index), [])  # 다른 규정은 안 봄

    def test_with_definitions_appends_one_block_and_skips_present(self):
        blocks = [{"ref": "보수규정 제29조", "doc_title": "보수규정", "score": 0.6, "text": "…"}]
        out = with_definitions("연봉월액은 얼마야?", blocks, self.index)
        self.assertEqual([b["ref"] for b in out], ["보수규정 제29조", "보수규정 제2조"])
        self.assertIn('"연봉월액"이란', out[1]["text"])
        self.assertEqual(out[1]["kind"], "definition")
        already = blocks + [{"ref": "보수규정 제2조", "doc_title": "보수규정", "score": 0.5, "text": "…"}]
        self.assertEqual(len(with_definitions("연봉월액은?", already, self.index)), 2)


if __name__ == "__main__":
    unittest.main()
