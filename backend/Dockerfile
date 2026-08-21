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
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip, setuptools, and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements.txt and install Python dependencies
COPY backend/backend-docvqa/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories for runtime storage (uploads, results, database)
RUN mkdir -p /app/uploads /app/results /app/db

# Copy application source code
COPY backend/backend-docvqa/backend /app
# Source code for the team's Stage 0-1 modules. Keeping it under /app makes
# `model.*` importable by Python; checkpoint files remain a runtime volume.
COPY model /app/model

# Create non-root user for security best practices and assign permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose backend port
EXPOSE 8000

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI backend with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
