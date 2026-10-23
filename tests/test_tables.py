import unittest

from ingestion.loaders import grid_to_blocks


def grid(rows):
    """rows: [[(text | None, colspan, rowspan), ...]] — None 은 병합으로 가려진 칸."""
    cells, spans, rowspans = {}, {}, {}
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            text, cs, rs = cell
            cells[(r, c)] = text.split("\n") if text else []
            spans[(r, c)], rowspans[(r, c)] = cs, rs
    return cells, spans, rowspans


class TwoRowHeaderTest(unittest.TestCase):
    # 위임전결표 축소판: 1행 "전결권자"(가로 병합) 아래 2행 "부장 · 부서장", "이사장" 은 세로 병합
    ROWS = [
        [("업무내용", 2, 1), None, ("전결권자", 2, 1), None, ("이사장", 1, 2)],
        [("단위업무", 1, 1), ("세부사항", 1, 1), ("부장", 1, 1), ("부서장", 1, 1), None],
        [("1. 문서관리", 1, 2), ("문서관리계획 수립", 1, 1), ("", 1, 1), ("○", 1, 1), ("", 1, 1)],
        [None, ("문서접수", 1, 1), ("○", 1, 1), ("", 1, 1), ("", 1, 1)],
        [("2. 예산", 1, 1), ("예산 편성", 1, 1), ("", 1, 1), ("", 1, 1), ("○", 1, 1)],
    ]

    def test_check_table_becomes_self_contained_lines(self):
        out = grid_to_blocks(*grid(self.ROWS))[0].splitlines()
        self.assertEqual(out, [
            "- 1. 문서관리 > 문서관리계획 수립 → 전결권자·부서장",
            "- 1. 문서관리 > 문서접수 → 전결권자·부장",        # 세로 병합된 "1. 문서관리" 가 채워짐
            "- 2. 예산 > 예산 편성 → 이사장",                  # 세로 병합 머리 "이사장" 은 한 단계 이름
        ])

    def test_non_check_table_keeps_markdown_with_merged_names(self):
        rows = [r[:] for r in self.ROWS]
        rows[2] = [("1. 문서관리", 1, 2), ("계획", 1, 1), ("가", 1, 1), ("나", 1, 1), ("다", 1, 1)]
        rows[3] = [None, ("접수", 1, 1), ("라", 1, 1), ("마", 1, 1), ("바", 1, 1)]
        rows[4] = [("2. 예산", 1, 1), ("편성", 1, 1), ("사", 1, 1), ("아", 1, 1), ("자", 1, 1)]
        lines = grid_to_blocks(*grid(rows))[0].splitlines()
        self.assertEqual(lines[0], "| 업무내용·단위업무 | 업무내용·세부사항 | 전결권자·부장 | 전결권자·부서장 | 이사장 |")
        self.assertEqual(lines[3], "| 1. 문서관리 | 접수 | 라 | 마 | 바 |")


class ExplodeRowTest(unittest.TestCase):
    def test_multi_line_cells_become_rows(self):
        rows = [[("직급", 1, 1), ("상한액", 1, 1)],
                [("1급\n2급\n3급", 1, 1), ("105,639,000\n92,598,000\n80,541,000", 1, 1)]]
        lines = grid_to_blocks(*grid(rows))[0].splitlines()
        self.assertEqual(lines[2:], ["| 1급 | 105,639,000 |", "| 2급 | 92,598,000 |", "| 3급 | 80,541,000 |"])

    def test_unequal_line_counts_are_joined(self):
        rows = [[("구분", 1, 1), ("내용", 1, 1)], [("가", 1, 1), ("첫 줄\n둘째 줄", 1, 1)]]
        self.assertEqual(grid_to_blocks(*grid(rows))[0].splitlines()[2], "| 가 | 첫 줄 둘째 줄 |")


if __name__ == "__main__":
    unittest.main()
