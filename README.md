# UVM Debug Agent

UVM (Universal Verification Methodology) 에러 분석 및 디버깅 도구입니다. 이 도구는 UVM 시뮬레이션 로그를 분석하여 에러의 근본 원인을 파악하고, 해결 방안을 제시합니다.

## 주요 기능

- UVM 로그 파일에서 에러 자동 수집
- SystemVerilog 코드 및 문서 분석
- RAG (Retrieval-Augmented Generation) 기반 에러 분석
- 상세한 분석 리포트 생성 (HTML, Markdown, YAML)
- 에러 패턴 및 트렌드 분석

## 설치 방법

1. 저장소 클론:
```bash
git clone https://github.com/yourusername/pcie_debug_agent.git
cd pcie_debug_agent
```

2. Docker 이미지 빌드:
```bash
docker build -t uvm-debug-agent .
```

3. 환경 변수 설정:
`.env` 파일을 생성하고 다음 환경 변수들을 설정합니다:
```bash
OPENAI_API_KEY=your_openai_api_key_here
SLACK_WEBHOOK_URL=your_slack_webhook_url_here
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_specific_password_here
```

## 사용 방법

1. 설정 파일 준비:
`configs/settings.yaml` 파일을 프로젝트에 맞게 수정합니다.

2. Docker 컨테이너 실행:
```bash
docker run --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/reports:/app/reports \
  uvm-debug-agent \
  --config configs/settings.yaml \
  --start-time "2024-03-20 00:00:00"
```

### 명령행 옵션

- `--config`: 설정 파일 경로 (기본값: configs/settings.yaml)
- `--start-time`: 분석 시작 시간 (YYYY-MM-DD HH:MM:SS 형식)
- `--log-level`: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## 프로젝트 구조

```
pcie_debug_agent/
├── configs/
│   └── settings.yaml
├── data/
│   ├── specs/
│   └── testbench/
├── src/
│   ├── collectors/
│   │   └── log_collector.py
│   ├── processors/
│   │   ├── code_chunker.py
│   │   ├── document_chunker.py
│   │   └── embedder.py
│   ├── vectorstore/
│   │   └── faiss_store.py
│   ├── rag/
│   │   ├── analyzer.py
│   │   └── retriever.py
│   ├── reports/
│   │   └── report_generator.py
│   └── main.py
├── templates/
│   ├── report.html.jinja2
│   └── report.markdown.jinja2
├── Dockerfile
├── requirements.txt
└── README.md
```

## 리포트 예시

도구는 다음 세 가지 형식의 리포트를 생성합니다:

1. HTML 리포트 (`reports/report.html`)
   - 인터랙티브 차트와 상세한 분석 결과
   - 에러 통계 및 트렌드 시각화

2. Markdown 리포트 (`reports/report.md`)
   - 간단한 텍스트 기반 리포트
   - 버전 관리 시스템에 적합

3. YAML 요약 (`reports/summary.yaml`)
   - 핵심 통계 및 분석 결과 요약
   - 자동화된 처리를 위한 구조화된 데이터

## 라이선스

MIT License

## 기여 방법

1. 이슈 생성 또는 기존 이슈 확인
2. 새로운 브랜치 생성
3. 변경사항 커밋
4. Pull Request 생성

## 문의

문제나 제안사항이 있으시면 이슈를 생성해 주세요.