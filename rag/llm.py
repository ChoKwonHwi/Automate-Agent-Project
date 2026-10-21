"""Ollama /api/chat 클라이언트 (표준 라이브러리만 사용)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from retrieval.config import ROOT, _load_dotenv


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    output_tokens: int
    total_ms: float
    load_ms: float


class OllamaChat:
    def __init__(self, model: str | None = None, host: str | None = None, num_ctx: int | None = None,
                 timeout: int = 300):
        env = {**_load_dotenv(ROOT / ".env"), **os.environ}
        self.model = model or env.get("LLM_MODEL", "qwen3:8b")
        self.host = (host or env.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.num_ctx = num_ctx or int(env.get("LLM_NUM_CTX", "8192"))
        self.timeout = timeout
        self.supports_think = True  # thinking 미지원 모델이면 첫 오류 후 False 로 바뀐다

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.host}{path}", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def chat(self, messages: list[dict], schema: dict | None = None, think: bool = False,
             temperature: float = 0.1) -> LLMResponse:
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            "keep_alive": "30m",
            "options": {"num_ctx": self.num_ctx, "temperature": temperature},
        }
        if self.supports_think:
            payload["think"] = think
        if schema:
            payload["format"] = schema
        try:
            data = self._post("/api/chat", payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if "think" in body and "think" in payload:  # thinking 옵션을 모르는 모델 → 빼고 재시도
                self.supports_think = False
                payload.pop("think")
                data = self._post("/api/chat", payload)
            else:
                raise RuntimeError(f"Ollama 호출 실패 ({self.model}): HTTP {exc.code} {body[:200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama 호출 실패 ({self.host}, {self.model}): {exc}") from exc
        return LLMResponse(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            total_ms=data.get("total_duration", 0) / 1e6,
            load_ms=data.get("load_duration", 0) / 1e6,
        )

    def warmup(self) -> float:
        """모델을 GPU 에 올린다 (첫 호출의 로드 시간을 지연 측정에서 빼기 위해). 로드 시간(ms) 반환."""
        resp = self.chat([{"role": "user", "content": "안녕"}], temperature=0.0)
        return resp.load_ms

    def unload(self) -> None:
        """GPU 에서 모델을 내린다 (모델 여러 개를 비교할 때 VRAM 8GB 초과 방지)."""
        try:
            self._post("/api/generate", {"model": self.model, "keep_alive": 0})
        except Exception:
            pass
