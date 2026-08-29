# ==============================================================================
# Document Visual Question Answering (DocVQA) System - Backend Service Dockerfile
# ==============================================================================
FROM python:3.11-slim

# Set environment variables for Python performance and unbuffered output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0

# Install system dependencies required for OpenCV, PaddleOCR, Pillow, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user first
RUN useradd -m -u 1000 appuser

# Upgrade pip tools with BuildKit cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install PyTorch CPU and dependencies with BuildKit cache
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Create storage directories
RUN mkdir -p /app/uploads /app/results && \
    chown -R appuser:appuser /app

# Copy backend source code
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
