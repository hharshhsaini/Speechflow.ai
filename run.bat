@echo off
setlocal enabledelayedexpansion

if not exist .env (
    echo Warning: .env file not found. Creating a default one.
    echo ANTHROPIC_API_KEY=your_key_here > .env
    echo WHISPER_MODEL=base >> .env
    echo DEFAULT_VOICE=af_heart >> .env
    echo HOST=0.0.0.0 >> .env
    echo PORT=8000 >> .env
)

findstr "your_key_here" .env >nul
if %errorlevel% equ 0 (
    echo WARNING: ANTHROPIC_API_KEY is not set in .env! The LLM module will not work.
)

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

echo Installing dependencies...
pip install -r requirements_agent.txt -q

echo Starting FastAPI server...
echo Open http://localhost:8000 in your browser
uvicorn server:app --reload --host 0.0.0.0 --port 8000
