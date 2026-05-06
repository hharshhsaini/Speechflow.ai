FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    WHISPER_MODEL=small

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    espeak-ng \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock requirements_agent.txt README.md LICENSE /app/
COPY speechflow /app/speechflow
COPY agent /app/agent
COPY static /app/static
COPY local_models /app/local_models
COPY server.py PROJECT_REFERENCE.md run.sh /app/

RUN pip install --upgrade pip setuptools wheel && \
    pip install .

EXPOSE 8000

CMD ["python", "server.py"]
