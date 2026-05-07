FROM python:3.12-slim

# Install system dependencies (espeak-ng required by misaki for G2P)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Use PORT env var from Render, default to 8000
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}
