import hashlib
import tempfile
import unittest
from pathlib import Path

from retrieval.lookup import lookup
from retrieval.refs import chunk_ref, normalize_ref, split_refs

try:
    import qdrant_client  # noqa: F401
except ImportError:
    qdrant_client = None


def chunk(cid: str, doc: str, section: str, label: str, text: str = "본문", part: int = 0) -> dict:
    return {"chunk_id": cid, "doc_id": doc, "org": "기관", "doc_title": doc, "category": "여비",
            "effective_date": "2024-02-02", "source_url": "", "section": section, "chapter": "",
            "article_label": label, "article_title": "", "part": part, "total_parts": 1,
            "page_start": 1, "page_end": 1, "text": text, "embed_text": f"[기관] {doc} > {label}\n{text}",
            "char_len": len(text)}


class RefsTest(unittest.TestCase):
    def test_chunk_ref_forms(self):
        self.assertEqual(chunk_ref(chunk("a", "여비규정", "본칙", "제16조")), "여비규정 제16조")
        self.assertEqual(chunk_ref(chunk("b", "여비규정 별표1", "별표", "별표 1")), "여비규정 별표1")
        self.assertEqual(chunk_ref(chunk("c", "여비규정", "별표", "별표 1의2")), "여비규정 별표1의2")
        self.assertEqual(chunk_ref(chunk("d", "민원처리예규 별표", "별표", "별표")), "민원처리예규 별표")
        self.assertEqual(chunk_ref(chunk("e", "여비규정", "부칙", "부칙2 제1조")), "여비규정 부칙2 제1조")

    def test_normalize_and_split(self):
        self.assertEqual(normalize_ref("여비규정 별표 1"), normalize_ref("여비규정 별표1"))
        self.assertEqual(split_refs("여비규정 별표1; 여비규정 제16조\n"), ["여비규정 별표1", "여비규정 제16조"])

    def test_lookup_collects_all_parts(self):
        chunks = [chunk("1", "여비규정", "본칙", "제16조", "①", 0), chunk("2", "여비규정", "본칙", "제16조", "②", 1),
                  chunk("3", "여비규정", "본칙", "제17조")]
        self.assertEqual([c["chunk_id"] for c in lookup(chunks, "여비규정  제16조")], ["1", "2"])


def fake_embed(texts):
    """텍스트 해시로 만든 4차원 단위벡터 (모델 없이 인덱서 로직만 검증)."""
    out = []
    for t in texts:
        h = hashlib.sha1(t.encode()).digest()
        v = [b + 1.0 for b in h[:4]]
        n = sum(x * x for x in v) ** 0.5
        out.append([x / n for x in v])
    return out


@unittest.skipIf(qdrant_client is None, "qdrant-client 미설치")
class IndexTest(unittest.TestCase):
    def setUp(self):
        from retrieval.index import ensure_collection, open_client

        self.tmp = tempfile.TemporaryDirectory()
        self.client = open_client(Path(self.tmp.name))
        ensure_collection(self.client, "t", 4)

    def tearDown(self):
        self.client.close()
        self.tmp.cleanup()

    def test_incremental_upsert_and_delete(self):
        from retrieval.index import index_chunks

        chunks = [chunk("a", "여비규정", "본칙", "제1조", "가"), chunk("b", "여비규정", "본칙", "제2조", "나")]
        quiet = lambda _: None
        self.assertEqual(index_chunks(self.client, "t", chunks, fake_embed, "m", log=quiet),
                         {"total": 2, "embedded": 2, "metadata": 0, "deleted": 0, "unchanged": 0})
        # 같은 입력 재실행 → 재임베딩 없음
        self.assertEqual(index_chunks(self.client, "t", chunks, fake_embed, "m", log=quiet)["unchanged"], 2)
        # b 본문 변경, a 삭제, c 추가
        changed = [chunk("b", "여비규정", "본칙", "제2조", "나나"), chunk("c", "여비규정", "본칙", "제3조", "다")]
        self.assertEqual(index_chunks(self.client, "t", changed, fake_embed, "m", log=quiet),
                         {"total": 2, "embedded": 2, "metadata": 0, "deleted": 1, "unchanged": 0})
        self.assertEqual(self.client.count("t").count, 2)

    def test_metadata_only_change_updates_payload_without_reembedding(self):
        from retrieval.index import index_chunks, point_id

        quiet = lambda _: None
        c = chunk("a", "여비규정", "본칙", "제1조", "가")
        index_chunks(self.client, "t", [c], fake_embed, "m", log=quiet)
        calls = []
        counting_embed = lambda texts: calls.append(texts) or fake_embed(texts)
        updated = {**c, "source_url": "https://law.go.kr/x"}
        stats = index_chunks(self.client, "t", [updated], counting_embed, "m", log=quiet)
        self.assertEqual((stats["embedded"], stats["metadata"]), (0, 1))
        self.assertEqual(calls, [])
        point = self.client.retrieve("t", [point_id("a")], with_payload=True)[0]
        self.assertEqual(point.payload["source_url"], "https://law.go.kr/x")

    def test_search_with_filter(self):
        from retrieval.index import index_chunks
        from retrieval.search import build_filter

        chunks = [chunk("a", "여비규정", "본칙", "제1조", "가"), chunk("b", "인사규정", "본칙", "제1조", "가")]
        index_chunks(self.client, "t", chunks, fake_embed, "m", log=lambda _: None)
        query = fake_embed([chunks[0]["embed_text"]])[0]
        points = self.client.query_points("t", query=query, limit=5, query_filter=build_filter(doc="인사규정"),
                                          with_payload=True).points
        self.assertEqual([p.payload["ref"] for p in points], ["인사규정 제1조"])


if __name__ == "__main__":
    unittest.main()
