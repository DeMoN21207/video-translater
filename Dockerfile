# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for moviepy/ffmpeg
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

# Install Python dependencies. Torch CPU build is used by default to keep the image lightweight.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

ARG WHISPER_MODELS="base"
ENV WHISPER_MODELS=${WHISPER_MODELS}

RUN if [ "${WHISPER_MODELS}" != "none" ] && [ -n "${WHISPER_MODELS}" ]; then \
        python - <<'PY'; \
import os

models = [m.strip() for m in os.environ.get("WHISPER_MODELS", "").split(",") if m.strip()]

if not models:
    raise SystemExit(0)

import whisper

for model_name in models:
    print(f"Pre-downloading Whisper model: {model_name}")
    whisper.load_model(model_name)
PY
    else \
        echo "Skipping Whisper model pre-download"; \
    fi

COPY . .

EXPOSE 5000

ENTRYPOINT ["python", "app.py"]
