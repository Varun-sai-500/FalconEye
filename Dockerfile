FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-gpu.txt .

RUN python -m pip install \
    --break-system-packages \
    -r requirements.txt

COPY . .