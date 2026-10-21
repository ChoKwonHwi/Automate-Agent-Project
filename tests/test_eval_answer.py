import unittest

from eval.answer import citation_scores, conclusion_ok, extract_numbers, grade, key_numbers, numeric_score, summarize
from eval.golden import GoldenItem
from rag.answer import AnswerResult


def item(type_="numeric", question="연봉이 60000000원이면?", answer="", evidence=("보수규정 제2조",)):
    return GoldenItem("q1", type_, "보수", question, answer, list(evidence))


class NumberTest(unittest.TestCase):
    def test_extract_units_and_ignore_article_numbers(self):
        self.assertEqual(extract_numbers("숙박비 10만원, 일비 25,000원 (여비규정 제16조, 별표 1)"), [100000, 25000])
        self.assertEqual(extract_numbers("약 166,666.67원"), [166666.67])
        self.assertEqual(extract_numbers("3천만원"), [30000000])
        self.assertEqual(extract_numbers("[시행 2024. 2. 2.]"), [])

    def test_key_numbers_exclude_question_numbers(self):
        i = item(answer="연봉 60,000,000원을 12로 나눈 5,000,000원, 하루 166,666.67원")
        self.assertEqual(key_numbers(i), [12, 5000000, 166666.67])

    def test_numeric_score_strict_rounding(self):
        i = item(answer="5,000,000원, 166,666.67원")
        self.assertEqual(numeric_score(i, "연봉월액 500만원, 일할 166,666.67원"), (1.0, []))
        score, missing = numeric_score(i, "연봉월액 5,000,000원, 일할 166,666원")
        self.assertEqual((score, missing), (0.5, [166666.67]))
        self.assertEqual(numeric_score(item(answer="해당 없음"), "x"), (None, []))

    def test_conclusion_number_is_lenient_on_conditions(self):
        i = item(question="3년 근무 후 전보되면 몇 년 제외?", answer="2년 이상 근무한 사람이 전보되면 1년간 제외됩니다.")
        self.assertEqual(numeric_score(i, "전보 후 1년간 제외")[0], 0.5)  # 엄격: 조건 숫자 2 누락
        self.assertTrue(conclusion_ok(i, "전보 후 1년간 제외"))          # 결론: 1년 일치


class GradeTest(unittest.TestCase):
    def test_citation_scores(self):
        self.assertEqual(citation_scores(["여비규정 별표 1", "여비규정 제9조"], ["여비규정 별표1", "여비규정 제16조"]),
                         {"hit": True, "precision": 0.5, "recall": 0.5})

    def test_grade_and_summarize(self):
        ok = AnswerResult("q", True, "5,000,000원", citations=["보수규정 제2조"], llm_ms=1000, prompt_tokens=100)
        refused = AnswerResult("q", False, "근거 없음", refused_by="llm", llm_ms=800, prompt_tokens=90)
        rows = [
            grade(item(answer="5,000,000원"), ok),
            {**grade(item(type_="fact", answer="답"), refused), "id": "q2"},
            {**grade(item(type_="refuse", answer="", evidence=()), refused), "id": "q3"},
        ]
        self.assertEqual((rows[0]["hit"], rows[0]["numeric"]), (True, 1.0))
        self.assertTrue(rows[1]["false_refusal"])
        self.assertTrue(rows[2]["correct_refusal"])
        s = summarize(rows)
        self.assertEqual((s["refusal_accuracy"], s["false_refusal_rate"], s["citation_hit"], s["numeric_exact"]),
                         (1.0, 0.5, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
