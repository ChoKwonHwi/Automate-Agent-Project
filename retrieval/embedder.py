"""BGE-M3 dense 임베딩 (sentence-transformers)."""
from __future__ import annotations

import json
import urllib.request
from functools import lru_cache

MAX_SEQ_LENGTH = 1024  # 청크는 1,500자 이하 → 토큰 1,000개 안팎. 길이를 묶어 VRAM 절약


@lru_cache(maxsize=2)
def _load(model_name: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}
    model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


class Embedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    @property
    def model(self):
        return _load(self.model_name, self.device)

    @property
    def dim(self) -> int:
        model = self.model
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return getter()

    def embed(self, texts: list[str], show_progress: bool = False) -> list[list[float]]:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=show_progress,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


def unload_ollama_models(host: str) -> list[str]:
    """GPU 메모리 확보를 위해 Ollama 에 올라가 있는 모델을 내린다 (PRD 8.3). 실패해도 무시."""
    unloaded = []
    try:
        with urllib.request.urlopen(f"{host}/api/ps", timeout=3) as resp:
            models = json.loads(resp.read()).get("models", [])
        for m in models:
            body = json.dumps({"model": m["name"], "keep_alive": 0}).encode()
            req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
            unloaded.append(m["name"])
    except Exception:
        pass
    return unloaded
