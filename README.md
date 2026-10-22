# Policy Copilot — 사내 규정 Q&A 및 업무 실행 Agent

임직원이 자연어로 사내 규정을 질문하면 **조항 단위 출처와 함께 답변**하고, 이어서 **신청서 초안 작성 · 규정 준수 검사 · 협업툴 등록**까지 수행하는 업무 Agent입니다.
외부 LLM API 없이 **로컬 GPU(Ollama)** 에서 동작하며, 모든 구성요소는 무료 · 오픈소스입니다.

- 상세 요구사항: [PRD.md](PRD.md)
- 데이터 수집 기준과 출처: [docs/data-collection.md](docs/data-collection.md)

## 주요 기능 (목표)
- 공공기관 실제 내부규정(PDF/HWP/HWPX)을 조·항 구조로 파싱해 인덱싱
- 하이브리드 검색(BM25 + BGE-M3) + Reranker 기반 RAG, 모든 답변에 조항 출처 인용
- LangGraph Agent: 질의 / 계산 / 신청서 작성 / 범위 외 분기
- MCP 서버(FastMCP)로 규정 검색 · 여비 계산 도구 제공, Notion · Slack 연동
- 외부 쓰기 액션은 사용자 승인 후 실행 (Human-in-the-loop)
- RAGAS 기반 평가, Langfuse 트레이싱

## 기술 스택
| 영역 | 사용 기술 |
|---|---|
| LLM | Ollama + Qwen3 8B (로컬 GPU, VRAM 8GB) |
| 임베딩 / 리랭커 | BGE-M3 / bge-reranker-v2-m3 |
| 검색 | Qdrant(dense) + BM25(Kiwi 형태소 분석), RRF 하이브리드 |
| Agent / 도구 | LangGraph, FastMCP |
| 서비스 | FastAPI, Streamlit, Docker Compose |
| 평가 / 관측 | RAGAS, Langfuse |

## 진행 현황
- [x] PRD 작성 및 확정
- [x] 프로젝트 구조, 문서 파서(PDF/HWP/HWPX), 조항 단위 청커, 검수 리포트
- [ ] W1 — 규정 데이터 수집 및 파싱 검수 (조항 경계 오류 < 5%)
  - 한국환경공단 규정 14종 · 46개 문서(본문 + 별표) 수집 · 파싱 완료, 청크 1,316개, 출처 URL 등록 완료
  - 남은 일: 경계 검수 샘플 30개 체크
- [ ] W2 — 인덱싱, 기본 RAG, Golden Set 100개, LLM 3종 비교 ([진행 상황](docs/plan-w2.md#진행-상황))
  - [x] BGE-M3(GPU) 임베딩 + Qdrant 증분 인덱서, 검색 · 조항 조회 CLI
  - [x] Golden Set 33문항 + 검색 평가 → **기준선 Recall@5 0.97 · MRR 0.873** ([리포트](eval/reports/w2_dense_baseline.md))
  - [x] 근거 기반 답변: 출처 인용 · 인용 검증 · 2단계 거절 · 정의 조항 자동 첨부
  - [x] 답변 자동 채점 + LLM 3종 비교 → **Qwen3 8B 확정** (인용 적중 86% · 거절 50%, [리포트](eval/reports/w2_llm_compare.md))
  - [ ] Golden Set 100문항으로 확장, 사람 판정 20개
  - 남은 과제: 거절 정확도(50% → 목표 90%)는 W3 quote 검증으로, 계산 오류는 W4 계산 도구로
- [ ] W3 — 하이브리드 검색 + Reranker + 거절 로직 ([진행 상황](docs/plan-w3.md#진행-상황))
  - [x] 인용문 대조 거절: **거절 50% → 100%**, 인용 적중 96% (오거절 14% — 남은 과제)
  - [x] 하이브리드 검색(dense + BM25, RRF): **Recall@5 0.95 → 1.00**, 기본 검색 방식으로 채택
  - [x] 일상어 → 규정 용어 사전 확장: R@1 0.88 → 0.92 (LLM 쿼리 재작성은 효과 없어 기본 끔)
  - [ ] 리랭커 · 두 줄 머리 표
- [ ] W4 — MCP 서버, LangGraph Agent, 계산 도구
- [ ] W5 — Notion/Slack 연동, 승인 흐름, API + UI
- [ ] W6 — Langfuse, Docker Compose, 평가 리포트

## 빠른 시작

### 1. 개발 환경
Python **3.11~3.13** 을 사용합니다 (PyTorch 등 ML 패키지 호환성 때문).

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

### 2. Ollama 점검
```bash
ollama pull qwen3:8b
python scripts/check_ollama.py
```
응답 속도(tokens/s), VRAM 사용량, JSON 구조화 출력 여부를 확인합니다.

### 3. 규정 문서 파싱
1. 규정 파일을 `data/raw/{기관약칭}/` 에 넣고 `data/sources.csv` 에 한 줄씩 등록합니다.
2. 파싱을 실행합니다.
   ```bash
   python -m ingestion.run
   ```
3. 결과
   - `data/processed/chunks.jsonl` — 조항 단위 청크
   - `data/processed/report.md` — 문서별 통계, 누락 조번호, 수작업 검수용 샘플

### 4. 임베딩 · 검색 (W2)
RTX 50 시리즈는 CUDA 12.8 빌드의 PyTorch 가 필요하다.
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[rag]"
```
```bash
python -m retrieval.index                      # 청크 → Qdrant(로컬 파일) 적재, 재실행 시 변경분만
python -m retrieval.search "서울 출장 숙박비 상한" -k 5 --category 여비   # --mode dense|bm25|hybrid
python -m retrieval.lookup "여비규정 제16조" "여비규정 별표1"   # 조항 원문 조회
python -m retrieval.lookup --list 여비규정                     # 참조 표기 목록
python -m eval.golden                                          # Golden Set 형식 검사 · 구성 현황
python -m eval.retrieval                                       # 검색 평가 → eval/reports/
python -m eval.answer --models qwen3:8b exaone3.5:7.8b        # 답변 평가 · 모델 비교 → eval/reports/
```

### 5. 근거 기반 답변 (W2)
```bash
ollama pull qwen3:8b
python -m rag.answer "대리가 서울로 출장 가면 숙박비 상한은?"            # 답변 + 출처
python -m rag.answer "결혼하면 휴가 며칠?" --show-context              # LLM 에 넣은 근거 원문도 출력
```
- 근거 밖 질문은 거절한다: 검색 점수가 매우 낮으면 즉시, 그 외에는 LLM 이 근거를 읽고 판단
- 답의 근거 문장(quotes)을 원문과 코드로 대조한다: 원문에 없거나 답의 기한 · 기간이 그 문장에 없으면 답하지 않는다
- 인용한 조항이 실제로 제공된 근거에 있는지 코드로 검증한다 (없는 조항 인용 시 경고)

## 디렉터리 구조
```
├─ PRD.md
├─ data/
│  ├─ sources.csv        # 수집 문서 출처 목록 (커밋 대상)
│  ├─ raw/               # 원본 규정 파일 (커밋 제외)
│  ├─ processed/         # 파싱 결과 (커밋 제외)
│  └─ eval/              # 평가 데이터셋
├─ ingestion/            # 문서 로더, 정제, 조항 청커, 리포트
├─ retrieval/            # 임베딩(BGE-M3), Qdrant 인덱스, 검색 · 조회 CLI
├─ rag/                  # 근거 기반 답변 (프롬프트, Ollama 클라이언트, 인용 검증 · 거절)
├─ eval/                 # Golden Set 검사, 검색 평가, 리포트(eval/reports)
├─ scripts/              # 환경 점검 스크립트
├─ mcp_servers/regulation/
├─ agent/  api/  ui/  eval/
└─ tests/
```

## 데이터 정책
- 원본 규정 파일과 가공 결과는 저장소에 포함하지 않습니다. `data/sources.csv` 의 출처 URL로 누구나 같은 데이터를 다시 받을 수 있습니다.
- AI Hub 데이터는 재배포가 금지되어 있어 저장소에 포함하지 않습니다.
- 평가 데이터셋(Golden Set)은 사람이 규정 원문을 보고 직접 작성하며, LLM으로 생성하지 않습니다.
