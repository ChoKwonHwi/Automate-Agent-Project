import tempfile
import unittest
from pathlib import Path

from ingestion.sources import load_sources

HEADER = "org,org_code,title,category,effective_date,source_url,collected_at,file_path,doc_id,notes\n"


def write_csv(text: str, encoding: str = "utf-8-sig") -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding=encoding, newline="")
    tmp.write(text)
    tmp.close()
    return Path(tmp.name)


class LoadSourcesTest(unittest.TestCase):
    def test_default_doc_id_and_cp949(self):
        path = write_csv(HEADER + "기관,org,여비규정,여비,2024-02-02,https://law.go.kr/a,2026-10-20,data/raw/org/여비규정.hwp,,\n",
                         encoding="cp949")
        self.assertEqual(load_sources(path)[0].doc_id, "org-여비규정")

    def test_url_in_doc_id_column_is_rejected(self):
        path = write_csv(HEADER + "기관,org,여비규정,여비,2024-02-02,TODO,2026-10-20,data/raw/org/여비규정.hwp,https://law.go.kr/a,\n")
        with self.assertRaisesRegex(ValueError, "source_url 칸"):
            load_sources(path)


if __name__ == "__main__":
    unittest.main()
