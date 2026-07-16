FROM pytorch/pytorch:2.12.1-cuda13.0-cudnn9-devel

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN python -m pip install \
    --break-system-packages \
    -r requirements-gpu.txt

COPY . .