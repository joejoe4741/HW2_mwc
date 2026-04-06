# --- Stage 1: Build & Dependencies ---
FROM python:3.9-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Pre-download DeepFace Models ---
FROM python:3.9-slim AS model-downloader
WORKDIR /app
# 💡 핵심: OpenCV 실행에 필요한 libglib 라이브러리를 설치합니다.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY main.py .
# 모델 다운로드 시 상세 로그를 보도록 PYTHONUNBUFFERED 설정
ENV PYTHONUNBUFFERED=1
RUN python -c "from deepface import DeepFace; import numpy as np; DeepFace.analyze(img_path=np.zeros((224, 224, 3), dtype=np.uint8), actions=['age', 'gender'], enforce_detection=False)"

# --- Stage 3: Final Runtime ---
FROM python:3.9-slim
WORKDIR /app
# 런타임에 필요한 라이브러리 (최적화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY --from=model-downloader /root/.deepface /root/.deepface
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
