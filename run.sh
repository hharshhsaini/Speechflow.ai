#!/bin/bash

# Check if .env exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating a default one."
    echo "ANTHROPIC_API_KEY=your_key_here" > .env
    echo "WHISPER_MODEL=base" >> .env
    echo "DEFAULT_VOICE=af_heart" >> .env
    echo "HOST=0.0.0.0" >> .env
    echo "PORT=8000" >> .env
fi

# Check for API key warning
if grep -q "your_key_here" .env; then
    echo "WARNING: ANTHROPIC_API_KEY is not set in .env! The LLM module will not work."
fi

# Run server using uv to ensure correct python version (3.12) and dependencies
echo "Starting FastAPI server using uv..."
echo "Open http://localhost:8000 in your browser"
uv run --python 3.12 --with soundfile --with pip uvicorn server:app --reload --host 0.0.0.0 --port 8000

