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
| 검색 | Qdrant + BM25(Kiwi 형태소 분석) |
| Agent / 도구 | LangGraph, FastMCP |
| 서비스 | FastAPI, Streamlit, Docker Compose |
| 평가 / 관측 | RAGAS, Langfuse |

## 진행 현황
- [x] PRD 작성 및 확정
- [x] 프로젝트 구조, 문서 파서(PDF/HWP/HWPX), 조항 단위 청커, 검수 리포트
- [ ] W1 — 규정 데이터 수집 및 파싱 검수 (조항 경계 오류 < 5%) — 한국환경공단 규정 45개 문서 수집 · 파싱 완료, 검수 진행 중
- [ ] W2 — 인덱싱, 기본 RAG, Golden Set 100개, LLM 3종 비교
- [ ] W3 — 하이브리드 검색 + Reranker + 거절 로직
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
python -m retrieval.search "서울 출장 숙박비 상한" -k 5 --category 여비
python -m retrieval.lookup "여비규정 제16조" "여비규정 별표1"   # 조항 원문 조회
python -m retrieval.lookup --list 여비규정                     # 참조 표기 목록
```

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
├─ scripts/              # 환경 점검 스크립트
├─ mcp_servers/regulation/
├─ agent/  api/  ui/  eval/
└─ tests/
```

## 데이터 정책
- 원본 규정 파일과 가공 결과는 저장소에 포함하지 않습니다. `data/sources.csv` 의 출처 URL로 누구나 같은 데이터를 다시 받을 수 있습니다.
- AI Hub 데이터는 재배포가 금지되어 있어 저장소에 포함하지 않습니다.
- 평가 데이터셋(Golden Set)은 사람이 규정 원문을 보고 직접 작성하며, LLM으로 생성하지 않습니다.
