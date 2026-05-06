# SpeechFlow.ai

SpeechFlow.ai is a local-first voice studio built with FastAPI, Whisper, and Kokoro/SpeechFlow TTS. It combines live dictation, multilingual voice synthesis, a voice testing lab, and a synced PDF book reader in a single web app.

## What the app does

- Live mic dictation into the studio workspace
- Neural TTS with 42 voice profiles across 10 languages
- Voice preview and health checks in Voice Lab
- Multilingual text-to-speech with translation to the selected voice language
- PDF reader with page view, read-along text, zoom, and word highlighting
- Local model execution with no external speech API required for the main app flow

## Stack

- Backend: FastAPI + WebSockets
- Frontend: vanilla HTML, CSS, and JavaScript
- STT: Whisper
- TTS: Kokoro / SpeechFlow local voices
- PDF parsing/rendering: PyMuPDF

## Repository layout

- `/server.py` - FastAPI app and websocket endpoints
- `/agent/tts.py` - synthesis, tokenization, translation, and voice registry
- `/agent/stt.py` - Whisper transcription helpers
- `/static/index.html` - full frontend UI
- `/local_models` - local Kokoro model weights, config, and voice packs

## Requirements

- Python 3.10 to 3.13
- `espeak-ng`
- `ffmpeg`
- `libsndfile`

Examples:

```bash
# macOS
brew install espeak-ng ffmpeg libsndfile

# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y espeak-ng ffmpeg libsndfile1
```

## Environment variables

Copy `.env.example` to `.env` and adjust as needed.

```env
HOST=0.0.0.0
PORT=8000
WHISPER_MODEL=small
DEFAULT_VOICE=af_heart
ANTHROPIC_API_KEY=your_key_here
```

Notes:

- `ANTHROPIC_API_KEY` is optional. It is only used by the separate Claude helper in `/agent/llm.py`, not by the main studio, reader, or voice lab flows.
- `WHISPER_MODEL=small` is the current default and the best balance here between latency and accuracy.

## Local development

Install Python dependencies:

```bash
pip install -r requirements_agent.txt
```

Run the app:

```bash
python server.py
```

Or use the helper script:

```bash
./run.sh
```

Then open:

```text
http://localhost:8000
```

Useful endpoints:

- `/` - main app
- `/health` - health check
- `/voices` - grouped voice registry

## Docker

This repo now includes a production-ready `Dockerfile`.

Build:

```bash
docker build -t speechflow-ai .
```

Run:

```bash
docker run --rm -p 8000:8000 --env-file .env speechflow-ai
```

Open:

```text
http://localhost:8000
```

## Deployment

This app should be deployed on a Docker-capable host or VM, not on static hosting and not on serverless platforms that do not support long-running Python processes, WebSockets, and local model files.

### Recommended deployment shape

Use a container host where you can:

1. Build from the included `Dockerfile`
2. Expose port `8000`
3. Set the environment variables from `.env.example`
4. Keep the process running continuously

### Generic deployment steps

1. Push this repository to GitHub.
2. Create a new Docker-based web service on your hosting platform.
3. Point it at this repository.
4. Build with the included `Dockerfile`.
5. Expose port `8000`.
6. Set:
   - `HOST=0.0.0.0`
   - `PORT=8000`
   - `WHISPER_MODEL=small`
   - optional `ANTHROPIC_API_KEY`
7. Deploy.
8. Verify with:

```text
https://your-domain/health
```

Expected response:

```json
{"status":"ok","pdf_loaded":false,"cached_pages":0}
```

## Deployment notes

- The app loads local speech models, so first boot is slower than later requests.
- PDF reader page turns are now cached and prewarmed, which improves later page loads substantially after upload.
- Performance depends heavily on the CPU and memory available on the host.
- If you want faster transcription or synthesis under load, deploy on a stronger machine rather than a minimal free-tier instance.

## Current feature set

- Studio
  - live dictation
  - text presets
  - neural token display
  - multilingual voice output
- Book Reader
  - upload PDF
  - synced read-along text
  - page navigation
  - in-panel zoom
  - word highlighting
- Voice Lab
  - preview every voice
  - quality grade display
  - health checks
  - batch audit

## License

MIT
