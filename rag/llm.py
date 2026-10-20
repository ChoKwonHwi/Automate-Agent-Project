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

    def chat(self, messages: list[dict], schema: dict | None = None, think: bool = False,
             temperature: float = 0.1) -> LLMResponse:
        payload = {
            "model": self.model, "messages": messages, "stream": False, "think": think,
            "keep_alive": "30m",
            "options": {"num_ctx": self.num_ctx, "temperature": temperature},
        }
        if schema:
            payload["format"] = schema
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama 호출 실패 ({self.host}, {self.model}): {exc}") from exc
        return LLMResponse(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            total_ms=data.get("total_duration", 0) / 1e6,
            load_ms=data.get("load_duration", 0) / 1e6,
        )
