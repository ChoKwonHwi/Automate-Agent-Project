import unittest

from ingestion.clean import clean_pages
from ingestion.loaders import Page
from ingestion.run import image_only_pages


class ImageOnlyPagesTest(unittest.TestCase):
    def test_detects_pages_with_images_but_no_text(self):
        pages = [
            Page(1, "규정\n제1조(목적) 이 규정은 목적을 정한다.\n법제처 1 국가법령정보센터"),
            Page(2, "규정\n법제처 2 국가법령정보센터", images=1),
            Page(3, "규정\n제2조(정의) 정의는 다음과 같다.\n법제처 3 국가법령정보센터", images=1),
        ]
        self.assertEqual(image_only_pages(pages, clean_pages(pages)), [2])


if __name__ == "__main__":
    unittest.main()
