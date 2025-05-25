# Python 3.9 기반 이미지 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 프로젝트 파일 복사
COPY requirements.txt .
COPY src/ src/
COPY configs/ configs/
COPY templates/ templates/

# Python 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 실행 명령
ENTRYPOINT ["python", "src/main.py"]
CMD ["--config", "configs/settings.yaml"] 