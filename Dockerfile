FROM pytorch/pytorch:2.14.0-cuda13.2-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements.txt .

RUN python -m pip install --break-system-packages --upgrade pip && \
    python -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]