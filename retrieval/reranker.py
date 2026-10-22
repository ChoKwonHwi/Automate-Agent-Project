"""cross-encoder 리랭커 (bge-reranker-v2-m3).

bi-encoder(BGE-M3)는 질문과 문서를 따로 벡터로 만들어 빠르지만, 질문과 문서의 단어를 서로 보지 못한다.
리랭커는 (질문, 문서)를 한 입력으로 넣어 관련도를 직접 판정한다 — 정확하지만 문서마다 모델을 돌려 느리므로
하이브리드 검색 후보 상위 20개만 다시 줄 세운다 (retrieve → rerank).

실측 (RTX 5060 Laptop, 후보 20개): CPU 83초 · GPU fp16 0.39초, VRAM +0.55GB (LLM · 임베딩과 함께 7.8GB / 8.15GB)
→ PRD 8.3 의 "CPU 실행" 계획을 GPU 로 바꿨다.
"""
from __future__ import annotations

from functools import lru_cache

MAX_LENGTH = 512   # 토큰. 긴 조문은 앞부분(조 제목 · ①항)이 남는다


@lru_cache(maxsize=2)
def _load(model_name: str, device: str):
    import torch
    from sentence_transformers import CrossEncoder

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    return CrossEncoder(model_name, device=device, max_length=MAX_LENGTH, model_kwargs=kwargs,
                        activation_fn=torch.nn.Sigmoid())  # 로짓 → 0~1 관련도


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "auto", batch_size: int = 8):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    @property
    def model(self):
        return _load(self.model_name, self.device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def warmup(self) -> None:
        self.score("워밍업", ["워밍업 문장"])
