import unittest

from rag.verify import normalize, quote_in_text, unsupported_periods, verify_quotes

ROW = "| 사내강사 | 일반강사 | 20,000 | 초과시간당 10,000원 추가 |"
ARTICLE = "제13조(민원문서의 처리) ② 지방조직은 이송 받은 민원문서를 처리할 때 필요한 경우 본사 관련 부서로 보고하고 지시를 받아야 한다."


class VerifyTest(unittest.TestCase):
    def test_normalize_ignores_spaces_pipes_and_dots(self):
        self.assertEqual(normalize("임ㆍ직원 | 20,000 |"), normalize("임·직원 20,000"))

    def test_quote_matching(self):
        self.assertTrue(quote_in_text("일반강사 20,000 초과시간당 10,000원 추가", ROW))  # 표 행을 파이프 없이 인용
        self.assertTrue(quote_in_text("지방조직은 이송받은 민원문서를 처리할 때 필요한 경우 본사 관련 부서로 보고하고 지시를 받아야 한다",
                                      ARTICLE))  # 띄어쓰기 차이
        self.assertTrue(quote_in_text("지방조직은 이송 받은 민원문서를 처리하는 때 필요한 경우 본사 관련 부서로 보고하고 지시를 받아야 한다",
                                      ARTICLE))  # 사소한 차이(할→하는) 허용
        self.assertFalse(quote_in_text("본사는 보고를 받은 날부터 8근무시간 이내에 지시하여야 한다", ARTICLE))
        self.assertFalse(quote_in_text("제13조", ARTICLE))  # 너무 짧음

    def test_verify_quotes_fixes_wrong_ref(self):
        blocks = [{"ref": "민원처리예규 제13조", "text": ARTICLE}, {"ref": "교육훈련규정 별표4", "text": ROW}]
        ok, bad = verify_quotes([{"ref": "민원처리예규 제13조", "text": "일반강사 20,000"},
                                 {"ref": "민원처리예규 제13조", "text": "본사는 8근무시간 이내에 지시한다"}], blocks)
        self.assertEqual(ok, [{"ref": "교육훈련규정 별표4", "text": "일반강사 20,000"}])
        self.assertEqual(len(bad), 1)

    def test_unsupported_periods(self):
        quotes = [{"ref": "A", "text": "승인통지를 받은 날부터 30일 이내에 수립하여야 한다"}]
        self.assertEqual(unsupported_periods("30일 이내에 수립해야 합니다", "며칠 이내?", quotes), [])
        self.assertEqual(unsupported_periods("8근무시간 이내에 지시해야 합니다", "며칠 안에?", quotes), ["8근무시간"])
        self.assertEqual(unsupported_periods("3년 근무했으면 1년간 제외", "3년 근무한 뒤 몇 년?", quotes), ["1년"])


if __name__ == "__main__":
    unittest.main()
