"""Ollama 점검: 설치 모델, 응답 속도, VRAM 적재 상태, JSON 구조화 출력.

    python scripts/check_ollama.py [--model qwen3:8b] [--runs 3]

표준 라이브러리만 사용하므로 가상환경 없이도 실행된다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "8192"))

PROMPT = "국내 출장 시 숙박비 상한을 정하는 규정의 일반적인 구성 요소를 세 문장으로 설명해줘."
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["qa", "calculate", "draft_form", "check_compliance", "out_of_scope"]},
        "reason": {"type": "string"},
    },
    "required": ["intent", "reason"],
}


def call(path: str, payload: dict | None = None, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(HOST + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def chat(model: str, content: str, **extra) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "think": False,
        "options": {"num_ctx": NUM_CTX},
        **extra,
    }
    return call("/api/chat", payload)


def rate(count: int, duration_ns: int) -> float:
    return count / (duration_ns / 1e9) if duration_ns else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "qwen3:8b"))
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    try:
        tags = call("/api/tags", timeout=5)
    except urllib.error.URLError as exc:
        print(f"[실패] Ollama 서버에 연결할 수 없습니다 ({HOST}): {exc}")
        return 1
    names = [m["name"] for m in tags.get("models", [])]
    print(f"[1] 설치된 모델: {names}")
    if args.model not in names:
        print(f"[실패] {args.model} 가 없습니다. `ollama pull {args.model}` 를 실행하세요.")
        return 1

    print(f"\n[2] 응답 속도 (num_ctx={NUM_CTX}, thinking off, {args.runs}회)")
    print("    회차 | 모델 로드(s) | 프롬프트 처리(tok/s) | 생성(tok/s) | 전체(s)")
    for i in range(1, args.runs + 1):
        r = chat(args.model, PROMPT)
        print(f"    {i:>4} | {r.get('load_duration', 0) / 1e9:>11.2f} | "
              f"{rate(r.get('prompt_eval_count', 0), r.get('prompt_eval_duration', 0)):>20.1f} | "
              f"{rate(r.get('eval_count', 0), r.get('eval_duration', 0)):>11.1f} | "
              f"{r.get('total_duration', 0) / 1e9:>7.2f}")

    print("\n[3] VRAM 적재 상태 (/api/ps)")
    for m in call("/api/ps").get("models", []):
        size, vram = m.get("size", 0), m.get("size_vram", 0)
        ratio = vram / size * 100 if size else 0
        print(f"    {m['name']}: 전체 {size / 2**30:.2f}GB, GPU {vram / 2**30:.2f}GB ({ratio:.0f}%), "
              f"context {m.get('context_length', '?')}")
        if ratio < 99:
            print("    [주의] 일부 레이어가 CPU에 올라가 있습니다. num_ctx 를 줄이거나 다른 GPU 프로그램을 종료하세요.")

    print("\n[4] JSON 구조화 출력 (의도 분류)")
    r = chat(args.model, "다음 주 화~목 부산 출장 신청서 만들어줘", format=INTENT_SCHEMA)
    try:
        parsed = json.loads(r["message"]["content"])
        print(f"    OK: {parsed}")
    except (KeyError, json.JSONDecodeError):
        print(f"    [실패] JSON 파싱 불가: {r.get('message', {}).get('content', '')[:200]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
