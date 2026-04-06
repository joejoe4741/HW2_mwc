# --- Stage 1: Build & Dependencies ---
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system dependencies for build (e.g., build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Pre-download DeepFace Models ---
# This ensures models are already present in the image
FROM python:3.9-slim AS model-downloader

WORKDIR /app
COPY --from=builder /install /usr/local
COPY main.py .

# Trigger model download (age prediction uses VGG-Face by default or specific models)
# Run a dummy script to force DeepFace to download weights
RUN python -c "from deepface import DeepFace; import numpy as np; DeepFace.analyze(img_path=np.zeros((224, 224, 3), dtype=np.uint8), actions=['age'], enforce_detection=False)"

# --- Stage 3: Final Runtime ---
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies (libgl1 for OpenCV if needed, though headless might not need it)
# opencv-python-headless doesn't need libGL, but let's keep it minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy pre-downloaded models from model-downloader
COPY --from=model-downloader /root/.deepface /root/.deepface

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
