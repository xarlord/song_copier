# =============================================================================
# Audio Generator Studio — Multi-stage Dockerfile for GPU cloud deployment
# Target: RunPod, Vast.ai, Lambda Labs, etc.
# CUDA 11.8 + PyTorch + Gradio on port 7860
# =============================================================================

# ---- Stage 1: Base image with CUDA, Python, system deps, and PyTorch --------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        libsndfile1-dev \
        sox \
        git \
        curl \
        wget \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install PyTorch with CUDA 11.8 support
RUN pip install --no-cache-dir --break-system-packages \
        torch>=2.1.0 torchaudio>=2.1.0 \
        --index-url https://download.pytorch.org/whl/cu118

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt \
    && rm -rf /tmp/requirements.txt

# ---- Stage 2: Application image ---------------------------------------------
FROM base AS app

# Labels for metadata
LABEL maintainer="Audio Generator Studio" \
      description="Local AI-powered audio generation studio with stem separation, voice swap, and song spin-offs" \
      version="1.0.0" \
      org.opencontainers.image.source="https://github.com/user/audio_generator"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Set working directory
WORKDIR /app

# Copy source code
COPY --chown=appuser:appuser . /app

# Create cache and output directories with proper permissions
RUN mkdir -p /app/cache /app/output /app/temp \
    && chown -R appuser:appuser /app/cache /app/output /app/temp

# Switch to non-root user
USER appuser

# Environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860 \
    PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/cache/huggingface \
    TORCH_HOME=/app/cache/torch \
    XDG_CACHE_HOME=/app/cache

# Expose Gradio port
EXPOSE 7860

# Health check — verify the Gradio server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fL http://localhost:7860/ || exit 1

# Entrypoint
CMD ["python", "app.py", "--port", "7860"]
