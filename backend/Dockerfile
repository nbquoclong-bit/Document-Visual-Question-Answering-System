# ==============================================================================
# Document Visual Question Answering (DocVQA) System - Backend Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies required for OpenCV, PyMuPDF, Pillow, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create non-root user and pre-create cache/app directories with proper permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /home/appuser/.cache/huggingface/hub /home/appuser/.EasyOCR/model /app/uploads /app/results /app/db && \
    chown -R appuser:appuser /home/appuser /app

# Ensure huggingface, torch, and easyocr use appuser cache directories
ENV HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/hub \
    TORCH_HOME=/home/appuser/.cache/torch \
    EASYOCR_MODULE_PATH=/home/appuser/.EasyOCR

# Upgrade pip, setuptools, and wheel with BuildKit pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Copy requirements.txt and install Python dependencies
# Pre-install torch & torchvision from the CPU wheel repo to avoid downloading 4+ GB of NVIDIA CUDA packages on Mac/CPU
COPY backend/backend-docvqa/backend/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy application source code with correct ownership
COPY --chown=appuser:appuser backend/backend-docvqa/backend /app
# Source code for the team's Stage 0-1 modules. Keeping it under /app makes
# `model.*` importable by Python; checkpoint files remain a runtime volume.
COPY --chown=appuser:appuser model /app/model

USER appuser

# Expose backend port
EXPOSE 8000

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI backend with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
