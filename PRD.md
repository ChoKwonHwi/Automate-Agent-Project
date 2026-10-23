# PRD — 사내 규정 Q&A 및 업무 실행 Agent (가칭: Policy Copilot)

| 항목 | 내용 |
|---|---|
| 문서 버전 | v1.0 |
| 작성일 | 2026-10-19 |
| 상태 | 확정 — 결정 내역은 [13. 확정 사항](#13-확정-사항) 참고 |
| 실행 환경 | 로컬 PC, NVIDIA RTX 5060 Laptop GPU (VRAM 8GB), Ollama 기반 로컬 LLM |
| 제약 | 전 구성요소 무료(오픈소스 / 무료 티어), 실제 공개 데이터 사용(합성 데이터 금지) |

---

## 1. 개요

### 1.1 한 줄 정의
임직원이 자연어로 사내 규정을 질문하면 **조항 단위 출처와 함께 답변**하고, 이어서 **신청서 초안 작성·규정 준수 검사·협업툴 등록**까지 수행하는 업무 Agent.

### 1.2 배경 / 문제
- 규정집(인사·복무·여비·회계·보수 등)은 수십~수백 페이지의 PDF/HWP로 흩어져 있고, 개정 이력 관리가 어렵다.
- 임직원은 담당 부서(인사·경영지원)에 반복 질문을 하고, 담당자는 같은 답을 반복한다.
- 규정을 확인한 뒤에도 신청서 작성·경비 처리 등 후속 업무는 별도 시스템에서 수작업으로 한다.

### 1.3 목표 (Goals)
| ID | 목표 | 측정 지표 |
|---|---|---|
| G1 | 규정 질문에 정확하고 근거 있는 답변 제공 | Faithfulness ≥ 0.85, 출처 인용률 100% |
| G2 | 원하는 조항을 빠르게 검색 | 검색 Recall@5 ≥ 0.85 |
| G3 | 규정 확인 → 후속 업무를 한 대화에서 완료 | 실행형 시나리오 성공률 ≥ 80% |
| G4 | 실서비스 수준 운영 | p95 응답 시간 ≤ 10초 (로컬 LLM 기준), 전 요청 트레이싱 |

### 1.4 비목표 (Non-Goals)
- 실제 그룹웨어/ERP 결재 시스템과의 연동 (MVP에서는 Notion·Slack으로 대체)
- 법률 자문 수준의 해석 제공 (규정 원문 근거 안내까지만)
- 규정 문서 작성·개정 기능

---

## 2. 사용자 (Persona)

| 페르소나 | 설명 | 주요 니즈 |
|---|---|---|
| 일반 임직원 | 규정을 잘 모르는 실무자 | "출장 일비가 얼마야?", "경조휴가 며칠이야?" 즉답 + 신청서 초안 |
| 경영지원/인사 담당자 | 반복 질문 응대자 | 반복 문의 감소, 경비 증빙 사전 검사 |
| 관리자(Admin) | 시스템 운영자 | 규정 문서 업로드·갱신, 접근 권한 관리, 품질 모니터링 |

---

## 3. 핵심 사용자 시나리오

**S1. 규정 질의응답**
> 사용자: "국내 출장 갈 때 숙박비 상한이 얼마야?"
> Agent: "여비규정 제12조 제2항에 따라 직급별 1박 상한은 … 입니다. [출처: ○○공단 여비규정 제12조, 2025-03-01 시행]"

**S2. 규정 확인 → 신청서 초안 → 등록**
> 사용자: "다음 주 화~목 부산 출장 신청서 만들어줘"
> Agent: 규정 기준 일비·숙박비 계산 → 신청서 초안 제시 → **사용자 승인** → Notion DB에 등록 + Slack 알림

**S3. 경비 규정 준수 검사 (Phase 2)**
> 사용자: 영수증 이미지 업로드
> Agent: OCR로 금액·일자·업종 추출 → 여비/회계 규정 대조 → "숙박비가 상한을 2만원 초과합니다 (제12조)" 경고

**S4. 근거 부족 시 거절**
> 사용자: "우리 회사 주가 전망 알려줘"
> Agent: "등록된 규정 문서에서 근거를 찾을 수 없습니다. 담당 부서: 경영지원팀"

---

## 4. 범위 (Scope)

### 4.1 MVP (Phase 1)
- 규정 문서 수집·파싱·조항 단위 인덱싱
- 하이브리드 검색 + Reranker 기반 RAG 답변 (조항 출처 인용)
- 의도 분류 Agent (질의 / 계산 / 신청서 작성 / 범위 외)
- MCP 도구: 규정 검색, 일비·숙박비 계산, Notion 등록, Slack 전송
- 실행형 액션 전 Human-in-the-loop 승인
- 웹 채팅 UI, 관리자 문서 업로드
- RAGAS 평가 파이프라인, Langfuse 트레이싱

### 4.2 Phase 2
- 영수증 OCR + 규정 준수 검사 (S3)
- 부서별 문서 접근 권한 (문서 레벨 ACL)
- Slack Bot 인터페이스 (Slack에서 직접 질문)
- 규정 개정 감지 및 변경 조항 요약 알림

### 4.3 Phase 3 (선택)
- 입찰공고 모니터링 Agent 등 타 업무 Agent를 같은 MCP 서버군에 추가

---

## 5. 기능 요구사항

### 5.1 문서 수집·인덱싱 (Ingestion)
| ID | 요구사항 | 우선순위 |
|---|---|---|
| F-ING-01 | PDF, HWP(→PDF/텍스트 변환), DOCX 업로드 지원 | P0 |
| F-ING-02 | 규정 구조(장·조·항·호) 인식 기반 청킹, 조항 경계에서 분할 | P0 |
| F-ING-03 | 청크 메타데이터: `기관명, 규정명, 조번호, 조제목, 시행일, 원문페이지, 문서ID` | P0 |
| F-ING-04 | 표(직급별 금액표 등)는 별도 청크 + 마크다운 테이블로 보존 | P0 |
| F-ING-05 | 동일 규정 재업로드 시 버전 관리 (최신 시행일 우선, 구버전 보관) | P1 |
| F-ING-06 | 인덱싱 결과 미리보기 (관리자가 청킹 결과 확인) | P1 |

### 5.2 검색·답변 (RAG)
| ID | 요구사항 | 우선순위 |
|---|---|---|
| F-RAG-01 | 하이브리드 검색: BM25(키워드) + Dense(BGE-M3), RRF로 결합 | P0 |
| F-RAG-02 | Reranker(bge-reranker-v2-m3)로 상위 k개 재정렬 | P0 |
| F-RAG-03 | 모든 답변에 출처(규정명·조번호·시행일) 인용, 클릭 시 원문 조항 표시 | P0 |
| F-RAG-04 | 검색 점수가 임계값 미만이면 "근거 없음" 응답 + 담당 부서 안내 | P0 |
| F-RAG-05 | 메타데이터 필터 (특정 규정·기관으로 범위 제한) | P1 |
| F-RAG-06 | 멀티턴 대화: 이전 대화 맥락으로 질문 재작성(query rewriting) | P1 |

### 5.3 Agent
| ID | 요구사항 | 우선순위 |
|---|---|---|
| F-AGT-01 | 의도 분류: `qa`, `calculate`, `draft_form`, `check_compliance`, `out_of_scope` | P0 |
| F-AGT-02 | 계산은 LLM이 아닌 **결정적 도구(Python)** 로 수행, 근거 조항과 함께 반환 | P0 |
| F-AGT-03 | 외부 쓰기 액션(Notion 등록, Slack 전송) 전 사용자 승인 필수 (interrupt) | P0 |
| F-AGT-04 | 도구 실패 시 재시도 1회 후 사용자에게 실패 사유 안내 | P0 |
| F-AGT-05 | 최대 스텝 수 제한 (무한 루프 방지, 기본 8) | P0 |
| F-AGT-06 | 사용자별 대화 세션 저장 및 이어하기 | P1 |

### 5.4 MCP 도구
| 도구명 | 서버 | 설명 | 유형 |
|---|---|---|---|
| `search_regulations(query, filters, top_k)` | 자체 (FastMCP) | 하이브리드 검색 결과 반환 | 읽기 |
| `get_article(doc_id, article_no)` | 자체 | 특정 조항 원문 조회 | 읽기 |
| `calc_travel_expense(grade, region, nights, days)` | 자체 | 일비·숙박비·식비 계산 | 계산 |
| `create_form_draft(form_type, fields)` | 자체 | 신청서 초안(JSON/Markdown) 생성 | 읽기 |
| `notion_create_page(db_id, properties)` | Notion MCP | 신청서 DB 등록 | **쓰기(승인 필요)** |
| `slack_post_message(channel, text)` | Slack MCP | 알림 전송 | **쓰기(승인 필요)** |
| `ocr_receipt(image)` | 자체 (Phase 2) | 영수증 항목 추출 | 읽기 |

### 5.5 UI
| ID | 요구사항 | 우선순위 |
|---|---|---|
| F-UI-01 | 채팅 화면: 답변 스트리밍, 출처 카드, 승인/거절 버튼 | P0 |
| F-UI-02 | 관리자 화면: 문서 업로드, 인덱싱 상태, 문서 목록 | P0 |
| F-UI-03 | 답변 피드백(👍/👎 + 사유) 수집 | P1 |
| F-UI-04 | 관리자 대시보드: 질문 수, 근거 없음 비율, 피드백 통계 | P2 |

---

## 6. 비기능 요구사항

| 영역 | 요구사항 |
|---|---|
| 비용 | 전 구성요소 무료. 유료 API는 사용하지 않거나 무료 티어 한도 내로 제한 |
| 성능 | p95 응답 ≤ 10초 (로컬 LLM), 첫 토큰 ≤ 3초 |
| 정확성 | 계산 결과는 결정적 코드로만 산출, LLM 생성 수치 금지 |
| 보안 | 프롬프트 인젝션 방어(문서 내 지시문 무시), 쓰기 도구 승인 필수, API 키는 `.env`로 관리 |
| 개인정보 | 로그에서 이름·전화번호·계좌번호 마스킹 |
| 관측성 | Langfuse로 전 요청 트레이싱 (검색 결과, 도구 호출, 토큰, 지연) |
| 이식성 | `docker compose up` 한 번으로 전체 기동 |
| 모델 교체 | LLM 공급자는 Ollama 고정이 기본. 모델명·호스트는 `.env` 설정만으로 교체 가능 (OpenAI 호환 엔드포인트 사용) |
| 자원 | 상시 VRAM 사용량 ≤ 7.5GB (8GB GPU 기준, 8.3 참고) |

---

## 7. 데이터

### 7.1 데이터 소스 (모두 실제 공개 데이터)
| 용도 | 소스 | 형식 | 비고 |
|---|---|---|---|
| 규정 코퍼스 | 국가법령정보센터 "공공기관 정관/규정" (보조: 기관 홈페이지 사규, ALIO 수시공시 내부규정) | PDF/HWP, Open API | 3개 기관의 인사·복무·여비·회계·보수 규정 수집 |
| 상위 법령 | 국가법령정보 Open API (open.law.go.kr) | XML/JSON | 근로기준법 등 규정이 참조하는 법령 |
| 평가셋 보조 | AI Hub 행정문서 기계독해 데이터 | JSON | 회원가입 후 무료, 이용약관 확인 필요 |
| 실제 질문 패턴 | 고용노동부·국세청 FAQ / 질의회시 | HTML | 평가 질문 작성 참고 |
| 영수증 (Phase 2) | SROIE, CORD | 이미지 + 라벨 | OCR 추출 정확도 평가용 |

### 7.2 데이터 처리 원칙
- 원본은 `data/raw/`에 보관하고, 파싱 결과는 `data/processed/`에 JSONL로 저장한다.
- 각 문서의 출처 URL·수집일·라이선스를 `data/sources.csv`에 기록한다.
- 여러 기관 규정을 섞어 쓰되, 메타데이터 `기관명`으로 구분한다. 데모 시 "가상의 회사" 하나로 묶는 필터를 제공한다.

### 7.3 평가 데이터셋 (Golden Set)
- 수집한 규정에서 직접 질문–정답–근거조항 쌍을 **최소 100개** 작성한다 (사람이 원문을 보고 작성; LLM 생성 금지).
- 구성: 단순 사실 50%, 표·수치 20%, 다중 조항 종합 15%, 근거 없음(거절해야 함) 15%.

---

## 8. 시스템 아키텍처

```
[Web UI (Streamlit 또는 Next.js)]
            │  HTTP / SSE
            ▼
[FastAPI 백엔드] ──── Langfuse (트레이싱)
            │
            ▼
[LangGraph Agent]
  ├─ intent_router ─┬─ qa ─────────► retrieve → rerank → generate(인용) → verify
  │                 ├─ calculate ──► search + calc tool → generate
  │                 ├─ draft_form ─► search + form tool → [human approval] → Notion/Slack
  │                 └─ out_of_scope ► 거절 응답
            │  MCP Client
            ▼
[MCP 서버들]
  ├─ regulation-mcp (FastMCP, 자체) : search / get_article / calc / form / ocr
  ├─ Notion MCP (공식)
  └─ Slack MCP
            │
            ▼
[저장소] Qdrant(벡터) · BM25 인덱스 · PostgreSQL 또는 SQLite(세션·피드백·문서 메타)
[모델]   Ollama (LLM: Qwen3 8B, 로컬 GPU) · BGE-M3 · bge-reranker-v2-m3
```

### 8.1 기술 스택 (기본안)
| 영역 | 선택 | 대안 |
|---|---|---|
| 언어 | Python 3.11+ | — |
| LLM | **Ollama + Qwen3 8B (Q4_K_M)** — 한국어·tool calling | 비교 후보: Gemma 3 (12B는 VRAM 초과로 4B만), EXAONE 3.5 7.8B |
| 임베딩 | BGE-M3 (sentence-transformers) | multilingual-e5-large |
| Reranker | bge-reranker-v2-m3 | — |
| 벡터DB | Qdrant (Docker) | Chroma, pgvector |
| 키워드 검색 | rank_bm25 + 한국어 형태소 분석(Kiwi) | Qdrant sparse vector |
| 문서 파싱 | PyMuPDF, Docling, HWP는 hwp5txt / LibreOffice 변환 | — |
| Agent | LangGraph | LlamaIndex Workflows |
| MCP | FastMCP (Python SDK) | — |
| 백엔드 | FastAPI | — |
| 프론트 | Streamlit (MVP) | Next.js (Phase 2) |
| DB | SQLite (MVP) → PostgreSQL | — |
| 평가 | RAGAS + 자체 Recall@k 스크립트 | — |
| 관측성 | Langfuse (self-host) | — |
| 배포 | Docker Compose (로컬 GPU 서버) | 데모 영상 녹화로 공유 (무료 클라우드에는 GPU가 없어 로컬 LLM 구동 불가) |

### 8.3 로컬 GPU 운용 계획 (RTX 5060 Laptop, VRAM 8GB)
| 구성요소 | 실행 위치 | 예상 VRAM | 비고 |
|---|---|---|---|
| Qwen3 8B Q4_K_M (Ollama) | GPU | **5.77GB (실측)** | 컨텍스트 8K 토큰으로 제한 (`num_ctx=8192`), GPU 적재 100% |
| BGE-M3 (질의 임베딩) | **CPU** (색인은 GPU) | 0 (색인 시 1.1GB) | W3 실측: 세 모델 모두 GPU 면 7.88GB 로 VRAM 이 RAM 으로 넘쳐 LLM 최대 10배 저하 → 질문 임베딩만 CPU(0.2~0.8초) |
| bge-reranker-v2-m3 | **GPU (fp16)** | **+0.55GB (실측)** | 상위 20개만 재정렬. W3 실측: CPU 83초 · GPU 0.39초 → GPU 채택. LLM · 임베딩과 함께 7.8GB / 8.15GB |
| 합계 (GPU) | | **7.29GB** (LLM + 리랭커) | 실측: 7.3GB 까지는 LLM 속도 유지, 7.88GB(세 모델 모두)는 한도 초과로 저하 |

**실측 성능 (2026-10-20, `scripts/check_ollama.py`, thinking off):** 생성 약 56 tokens/s, 프롬프트 처리 약 2,800 tokens/s, 모델 상주 시 짧은 응답 1.6~1.9초. 콜드 스타트(첫 요청)는 약 9초 → `OLLAMA_KEEP_ALIVE` 로 상주 유지 필요.

- **Ollama 설정:** `OLLAMA_KEEP_ALIVE=30m`(모델 상주로 첫 응답 지연 제거), `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`
- **프롬프트 예산 (8K):** 시스템 프롬프트 ~1K + 검색 청크 최대 5개 ~4K + 대화 이력 ~1.5K + 생성 ~1.5K
- **Qwen3 thinking 모드:** 의도 분류·도구 호출은 비활성(`/no_think`), 다중 조항 종합 답변만 활성
- **동시 사용자:** MVP는 1~3명 수준을 전제로 하며, 요청은 큐로 순차 처리
- **Docker:** Ollama는 Windows 호스트에 직접 설치하고, 컨테이너에서는 `host.docker.internal:11434`로 접근 (WSL2 GPU 패스스루 이슈 회피)
- **모델 비교 실험 (W2):** Qwen3 8B / EXAONE 3.5 7.8B / Gemma 3 4B를 Golden Set으로 비교해 최종 모델 확정

### 8.2 디렉터리 구조 (안)
```
project/
├─ PRD.md
├─ docker-compose.yml
├─ .env.example
├─ data/
│  ├─ raw/  processed/  eval/
│  └─ sources.csv
├─ ingestion/          # 수집·파싱·청킹·인덱싱
├─ mcp_servers/
│  └─ regulation/      # FastMCP 서버
├─ agent/              # LangGraph 그래프, 노드, 프롬프트
├─ api/                # FastAPI
├─ ui/                 # Streamlit
├─ eval/               # RAGAS, Recall@k
└─ tests/
```

---

## 9. API (초안)
| Method | Path | 설명 |
|---|---|---|
| POST | `/chat` | 메시지 전송, SSE 스트리밍 응답 (`answer`, `citations`, `pending_action`) |
| POST | `/chat/{session_id}/approve` | 대기 중인 쓰기 액션 승인/거절 |
| GET | `/sessions/{session_id}` | 대화 이력 조회 |
| POST | `/admin/documents` | 규정 문서 업로드 및 인덱싱 |
| GET | `/admin/documents` | 문서 목록·인덱싱 상태 |
| POST | `/feedback` | 답변 피드백 저장 |
| GET | `/health` | 헬스체크 |

---

## 10. 평가 계획

| 구분 | 지표 | 목표 (MVP) | 방법 |
|---|---|---|---|
| 검색 | Recall@5, MRR | ≥ 0.85, ≥ 0.70 | Golden Set의 근거조항 기준 |
| 생성 | Faithfulness | ≥ 0.85 | RAGAS |
| 생성 | Answer Relevancy | ≥ 0.80 | RAGAS |
| 인용 | 인용 정확도 | ≥ 90% | 인용 조항이 실제 근거 조항과 일치하는 비율 |
| 거절 | 근거 없음 질문 정답 거절률 | ≥ 90% | Golden Set 거절 항목 |
| 계산 | 계산 정확도 | 100% | 도구 단위 테스트 |
| Agent | 시나리오 성공률 (S2) | ≥ 80% | 20개 실행형 시나리오 수동/자동 검증 |
| 성능 | p95 지연 | ≤ 10초 | Langfuse |

**실험 기록:** 청킹 방식(고정 길이 vs 조항 단위), 검색 방식(Dense vs Hybrid vs Hybrid+Rerank)별 지표를 비교표로 남긴다.

---

## 11. 마일스톤

| 주차 | 산출물 | 완료 기준 |
|---|---|---|
| W1 | 데이터 수집, 파서, 조항 단위 청킹 | 3개 기관 규정 JSONL 변환, 조항 경계 오류 < 5% |
| W2 | 인덱싱 + 기본 RAG + Golden Set 100개 + LLM 3종 비교 | 출처 인용 답변, 베이스라인 지표 측정, 최종 LLM 확정 |
| W3 | 하이브리드 검색 + Reranker + 거절 로직 | Recall@5 ≥ 0.85 |
| W4 | regulation-mcp 서버, LangGraph Agent, 계산 도구 | S1, S4 시나리오 통과 |
| W5 | Notion/Slack MCP 연동, 승인 흐름, FastAPI + UI | S2 시나리오 통과 |
| W6 | Langfuse, Docker Compose, 평가 리포트, 데모 영상 | 로컬 GPU에서 `docker compose up`으로 전체 기동, 지표 목표 달성 여부 리포트 |
| Phase 2 | OCR 경비 검사, ACL, Slack Bot | 별도 계획 |

---

## 12. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| HWP 파싱 품질 저하 | 청킹·검색 정확도 하락 | PDF 버전 우선 수집, LibreOffice 변환 fallback, 관리자 미리보기 |
| 로컬 LLM의 한국어·tool calling 품질 | 오답, 도구 오호출 | W2에 모델 3종 비교, 의도 분류는 JSON 구조화 출력(Ollama `format`)으로 강제 |
| VRAM 8GB 부족 (OOM, 속도 저하) | 응답 지연, 크래시 | 8.3 예산 준수, reranker CPU 이동, 컨텍스트 8K 제한, 필요 시 Qwen3 4B로 하향 |
| 노트북 GPU 발열·스로틀링 | 장시간 운용 시 지연 증가 | 전원 연결·고성능 모드 운용, Langfuse로 지연 추이 모니터링 |
| 기관마다 규정 내용 상이 | 답변 혼선 | `기관명` 필터를 기본 적용, 답변에 기관명 명시 |
| 문서 내 프롬프트 인젝션 | 의도치 않은 도구 실행 | 검색 결과는 데이터로만 취급하는 시스템 프롬프트, 쓰기 도구 승인 필수 |
| 규정 해석 오류의 업무 영향 | 잘못된 신청·지급 | "최종 확인은 담당 부서" 고지, 원문 조항 링크 상시 제공 |

---

## 13. 확정 사항

| # | 항목 | 결정 |
|---|---|---|
| D1 | LLM 실행 방식 | **Ollama 로컬 LLM 기본** (RTX 5060 Laptop 8GB). **Qwen3 8B Q4_K_M 확정 (2026-10-21, W2 비교)** — EXAONE 3.5 7.8B · Gemma 3 4B 대비 거절 · 인용 정밀도 우위, tool calling 지원. 결과: `eval/reports/w2_llm_compare.md`. 외부 LLM API는 사용하지 않음 |
| D2 | 데모용 "가상 회사" 기준 기관 | 규정이 잘 정리된 공공기관 1곳을 주 기관으로, 보조 기관 2곳 추가 (W1 데이터 수집 시 선정) |
| D3 | UI | MVP는 Streamlit, Phase 2에서 Next.js 검토 |
| D4 | 협업툴 연동 | Notion을 신청서 등록용으로 우선 구현, Slack은 알림 전송만 |
| D5 | 개발 체제·일정 | 1인 개발, MVP 6주 |
| D6 | Phase 2 OCR | 로컬 OCR(PaddleOCR) + 규칙 기반 검사 (8GB VRAM에서 멀티모달 LLM 동시 구동 부담) |

### 13.1 사전 준비 (W1 시작 전)
- [ ] Ollama Windows 설치 및 `ollama pull qwen3:8b` (현재 PC에 미설치 확인)
- [ ] NVIDIA 드라이버 최신화 (RTX 50 시리즈 CUDA 지원 확인)
- [ ] Docker Desktop (WSL2 백엔드) 설치
- [ ] Python 3.11+ 및 가상환경 구성
- [ ] Notion Integration 생성 및 신청서 DB 준비, Slack 무료 워크스페이스·Bot 토큰 발급
- [ ] 국가법령정보 Open API 키 신청, AI Hub 회원가입

---

## 14. 용어
- **RAG**: 검색 결과를 LLM 입력에 넣어 근거 기반으로 답변을 생성하는 방식
- **MCP (Model Context Protocol)**: LLM 애플리케이션이 외부 도구·데이터에 표준 방식으로 접근하게 하는 프로토콜
- **Hybrid Search**: 키워드(BM25)와 의미(벡터) 검색을 결합하는 방식
- **RRF (Reciprocal Rank Fusion)**: 여러 검색 결과의 순위를 합치는 방법
- **Human-in-the-loop**: 외부 영향이 있는 액션을 사람이 승인한 뒤 실행하는 구조
- **Golden Set**: 사람이 만든 정답 기반 평가 데이터셋
