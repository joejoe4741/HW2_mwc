# --- Stage 1: Build & Dependencies ---
FROM python:3.9-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Pre-download DeepFace Models (Robust Version) ---
FROM python:3.9-slim AS model-downloader
WORKDIR /app
# 💡 핵심: OpenCV 실행 및 모델 로딩에 필요한 모든 시스템 라이브러리를 설치합니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY main.py .
ENV PYTHONUNBUFFERED=1
# 💡 팁: 모델을 바로 실행하지 않고, Import 테스트와 함께 모델을 로드만 시도합니다.
# 만약 여전히 실패한다면 아래 RUN 문을 주석 처리(#)하고 빌드해 보세요.
RUN python -c "from deepface import DeepFace; import numpy as np; \
    print('Loading models...'); \
    DeepFace.build_model('VGG-Face'); \
    DeepFace.build_model('Age'); \
    DeepFace.build_model('Gender'); \
    print('Models pre-downloaded successfully.')"

# --- Stage 3: Final Runtime ---
FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY --from=model-downloader /root/.deepface /root/.deepface
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
