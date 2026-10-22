import json
import unittest

from rag.llm import LLMResponse
from retrieval.glossary import Entry, expand, load_glossary
from retrieval.rewrite import QueryRewriter

# 받침이 붙으면 다른 음절이 되므로("그만둔" 은 "그만두" 로 시작하지 않는다) 활용형을 따로 적는다
ENTRIES = (Entry(("그만두", "그만둔", "퇴사"), "퇴직 면직"), Entry(("회사차",), "공단차량"), Entry(("얘기",), "누설 비밀"))


class GlossaryTest(unittest.TestCase):
    def test_expand_ignores_spacing_and_collects_all(self):
        self.assertEqual(expand("그만둔 다음에 회사 일 얘기해도 돼?", ENTRIES), ["퇴직 면직", "누설 비밀"])
        self.assertEqual(expand("회사 차 타고 출장", ENTRIES), ["공단차량"])
        self.assertEqual(expand("점심 메뉴 추천", ENTRIES), [])

    def test_shipped_glossary_terms_appear_in_regulations(self):
        """사전이 넓혀 주는 규정 용어는 실제 원문에 있어야 검색에 쓸모가 있다."""
        from retrieval.config import get_settings
        from retrieval.index import load_chunks

        path = get_settings().chunks_path
        if not path.exists():
            self.skipTest("chunks.jsonl 없음")
        corpus = "".join(c["text"] for c in load_chunks(path)).replace(" ", "")
        entries = load_glossary()
        self.assertGreater(len(entries), 20)
        missing = [t for e in entries for t in e.expansion.split() if t.replace(" ", "") not in corpus]
        self.assertEqual(missing, [], f"원문에 없는 확장 용어: {missing}")


class FakeLLM:
    def __init__(self, content):
        self.content = content

    def chat(self, messages, schema=None, think=False, temperature=0.1):
        if isinstance(self.content, Exception):
            raise self.content
        return LLMResponse(self.content, 10, 10, 100.0, 0.0)


class RewriterTest(unittest.TestCase):
    def test_parses_and_caps_queries(self):
        content = json.dumps({"queries": ["a", "b", "c", "d", " "], "hypothetical": "조문"})
        rw = QueryRewriter(FakeLLM(content)).rewrite("질문")
        self.assertEqual((rw.queries, rw.hypothetical, rw.empty), (["a", "b", "c"], "조문", False))

    def test_failure_falls_back_to_empty(self):
        self.assertTrue(QueryRewriter(FakeLLM(RuntimeError("down"))).rewrite("질문").empty)
        self.assertTrue(QueryRewriter(FakeLLM("not json")).rewrite("질문").empty)


if __name__ == "__main__":
    unittest.main()
