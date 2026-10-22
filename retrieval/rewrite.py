"""쿼리 재작성: 일상어 질문을 규정 용어로 바꿔 검색을 보강한다.

"그만둔 다음에 회사 일 얘기해도 돼?" 의 정답(취업규정 제5조 "직원이었던 자는 직무상 알게 된 사항을 누설하여서는
아니 된다")은 겹치는 단어가 없어 dense · BM25 모두 상위 30개 밖이었다 (W3 하이브리드 검색 평가).

LLM 한 번 호출로 두 가지를 만든다.
- queries      : 질문을 서로 다르게 해석한 규정 용어 검색어 2~3개 → 각각 hybrid 검색
                 (일상어는 해석이 갈린다: "회사 일 얘기해도 돼?" = 업무 협의? 비밀 누설? — 하나에 걸면 빗나간다)
- hypothetical : 규정 말투로 쓴 가상 조문 (HyDE) → dense 검색
가상 조문은 지어낸 문장이므로 **검색에만** 쓰고 답변 근거에는 절대 넣지 않는다.
원래 질문 검색은 항상 함께 하므로 재작성이 빗나가도 기존 결과는 사라지지 않는다.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

REWRITE_SYSTEM = """너는 사내 규정 검색을 돕는다. 직원의 일상적인 질문을 규정 원문에 나올 법한 용어로 바꾼다.
- queries: 질문을 규정 관점에서 **서로 다르게 해석한** 검색어 2~3개. 각 검색어는 규정 원문에 쓰일 명사 위주 3~8개 단어.
  해석이 하나뿐이면 1개도 된다. 예: 구어 "월급" → "보수 기준급 연봉".
- hypothetical: 이 질문에 답하는 규정 조문이 있다면 어떻게 쓰여 있을지 규정 말투로 한두 문장.
  숫자 · 기한 · 금액은 모르면 쓰지 않는다. 사실 여부와 관계없이 검색용 예시 문장이다.
- 회사 생활(근무 · 휴가 · 보수 · 복리후생 · 출장 · 인사 · 퇴직 · 징계 · 윤리 · 계약 · 예산 · 감사 · 민원 등)과
  조금이라도 관련 있으면 반드시 채운다. 구어체("그만두다", "회사 일", "쉬다")도 규정 용어로 바꿔 채운다.
  판단이 애매하면 채운다 — 답할 수 있는지는 뒤 단계에서 따로 판단한다.
- 회사 생활과 전혀 관계없는 질문(날씨, 음식 추천, 연예 등)만 queries 를 빈 목록, hypothetical 을 빈 문자열로 둔다.
JSON 으로만 답한다."""

# 평가 문항과 겹치지 않는 예시만 쓴다 (예시가 평가 질문의 정답을 알려 주면 측정이 무의미해진다)
FEW_SHOT = [
    ("월급은 매달 언제 들어와?",
     {"queries": ["보수 지급일 지급 시기 매월"],
      "hypothetical": "보수는 매월 정해진 지급일에 직원의 예금계좌에 입금하여 지급한다."}),
    ("회사 카드로 팀 회식비 내도 돼?",
     {"queries": ["법인카드 사용 제한 업무추진비 집행", "회식 경비 예산 집행 기준"],
      "hypothetical": "업무추진비는 법인카드로 집행하며, 업무와 관련 없는 용도로 사용하여서는 아니 된다."}),
    ("애 키우느라 잠깐 회사 쉬고 싶은데 돼?",
     {"queries": ["육아휴직 자녀 양육 휴직기간", "육아기 근로시간 단축", "특별휴가 자녀 돌봄"],
      "hypothetical": "직원이 만 8세 이하의 자녀를 양육하기 위하여 필요한 경우에는 휴직을 명할 수 있다."}),
    ("오늘 저녁 뭐 먹지?", {"queries": [], "hypothetical": ""}),
]

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}, "hypothetical": {"type": "string"}},
    "required": ["queries", "hypothetical"],
}
MAX_QUERIES = 3


@dataclass
class Rewrite:
    queries: list[str]
    hypothetical: str
    ms: float

    @property
    def empty(self) -> bool:
        return not (self.queries or self.hypothetical.strip())


def build_messages(question: str) -> list[dict]:
    messages = [{"role": "system", "content": REWRITE_SYSTEM}]
    for q, a in FEW_SHOT:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    return messages + [{"role": "user", "content": question}]


class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, question: str) -> Rewrite:
        started = time.perf_counter()
        try:
            resp = self.llm.chat(build_messages(question), schema=REWRITE_SCHEMA, temperature=0.0)
            data = json.loads(resp.content)
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()][:MAX_QUERIES]
            hyp = str(data.get("hypothetical", "")).strip()
        except Exception:
            queries, hyp = [], ""  # 재작성 실패는 원래 질문 검색만으로 진행
        return Rewrite(queries, hyp, (time.perf_counter() - started) * 1000)
