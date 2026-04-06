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
COPY --from=builder /install /usr/local
COPY main.py .
# Pre-download models by running a dummy analysis
RUN python -c "from deepface import DeepFace; import numpy as np; DeepFace.analyze(img_path=np.zeros((224, 224, 3), dtype=np.uint8), actions=['age', 'gender'], enforce_detection=False)"

# --- Stage 3: Final Runtime ---
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
# Copy pre-downloaded models from model-downloader stage
COPY --from=model-downloader /root/.deepface /root/.deepface
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
