import csv
import io
import unittest

from eval.golden import check
from retrieval.refs import normalize_ref

KNOWN = {normalize_ref(r) for r in ["여비규정 별표1", "여비규정 제16조", "인사규정세칙 제57조"]}
HEADER = "id,type,category,question,answer,evidence,notes\n"


def rows(text: str):
    return list(enumerate(csv.DictReader(io.StringIO(HEADER + text)), start=2))


class GoldenCheckTest(unittest.TestCase):
    def test_valid_rows(self):
        items, problems = check(rows(
            'q1,numeric,여비,숙박비?,"25,000원, 100,000원",여비규정 별표 1,\n'
            "q2,multi,여비,차 타면?,절반,여비규정 별표1; 여비규정 제16조,\n"
            "q3,refuse,기타,주식?,,,\n"), KNOWN)
        self.assertEqual(problems, [])
        self.assertEqual(items[1].evidence, ["여비규정 별표1", "여비규정 제16조"])

    def test_unquoted_comma_detected(self):
        _, problems = check(rows("q1,numeric,여비,숙박비?,각각 25,000원,여비규정 별표1,\n"), KNOWN)
        self.assertIn("큰따옴표", problems[0])

    def test_unknown_ref_and_rules(self):
        _, problems = check(rows(
            "q1,numeric,여비,질문,답,여비규정 별표 1 비고 제3호,\n"
            "q1,multi,여비,질문,답,여비규정 제16조,\n"
            "q3,refuse,기타,질문,,여비규정 제16조,\n"
            "q4,사실,여비,질문,답,여비규정 제16조,\n"), KNOWN)
        text = "\n".join(problems)
        for expected in ("찾을 수 없습니다", "id 중복", "2개 이상", "evidence 를 비워", "type 은"):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
