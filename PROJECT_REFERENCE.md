# 🚀 SpeechFlow.ai Pro Neural Studio | Technical Handover Guide

This document provides a high-level architectural overview and technical implementation details for the **SpeechFlow.ai Pro Neural Studio**. Use this as a reference for extending the platform or debugging the core neural engines.

---

## 🏗️ 1. System Architecture

SpeechFlow.ai is a **local-first, multimodal AI agent** built for high-fidelity voice synthesis and real-time document processing.

- **Backend**: FastAPI (Python 3.10+) running on Uvicorn.
- **Frontend**: Vanilla HTML5 / CSS3 / JavaScript (No frameworks). High-performance UI with CSS Grid and glassmorphism.
- **Neural Engine**: 
  - **TTS**: [Kokoro](https://github.com/hexgrad/kokoro) (Local weights, 24kHz, 54+ voices).
  - **STT**: [OpenAI Whisper](https://github.com/openai/whisper) (Small model, low latency).
  - **PDF Engine**: [PyMuPDF (fitz)](https://github.com/pymupdf/PyMuPDF) for native text/image extraction.

---

## 📂 2. Core File Structure

| File | Responsibility |
| :--- | :--- |
| `server.py` | Main API server. Manages WebSockets for real-time audio/text streaming. |
| `agent/tts.py` | Centralized **VOICE_REGISTRY**. Manages pipeline caching and language fallback logic. |
| `static/index.html` | The entire "Pro Suite" UI. Contains Studio, Book Reader, and Voice Lab logic. |
| `requirements_agent.txt` | Core neural dependencies (torch, transformers, misaki, etc.). |

---

## ⚡ 3. Critical Implementation Details

### A. The "Smooth Reading" Engine (`server.py`)
To avoid robotic pauses, the Book Reader uses **sentence-level chunking**.
- **Logic**: Text is split at `[. ! ?]` boundaries using natural punctuation detection. 
- **Streaming**: Each sentence is synthesized into a WAV chunk and sent via WebSocket.
- **Timing**: Precise `word_index` and `start_time/end_time` offsets are calculated per chunk and broadcasted to the UI for synchronization.

### B. Highlighting Sync (`index.html`)
The frontend uses a **cumulative offset system** to keep highlights in sync across multiple audio chunks.
- `chunkDurationOffset`: Stores the total duration of all finished audio chunks.
- `readerLoop`: Runs at 60fps using `requestAnimationFrame`.
- `totalElapsed = chunkDurationOffset + readerAudio.currentTime`.

### C. Voice Lab & Registry (`agent/tts.py`)
Supports **54 profiles** across American English, British English, Japanese, Chinese, and Hindi.
- **Pipelines**: Cached by `lang_code` to prevent memory exhaustion when switching languages.
- **Health Checks**: `/voices/test/{id}` verifies synthesis path without generating full audio.

---

## 🛠️ 4. Local Environment Requirements

1. **System Deps**: `espeak-ng` (Required by Kokoro for phonemization).
2. **Python Environment**: Managed via `uv` or standard virtual environment.
3. **NLP Models**: 
   - NLTK `punkt` for sentence splitting.
   - spaCy `en_core_web_sm` for English neural mapping.
4. **Execution**: `uv run server.py` (Runs on `http://localhost:8000`).

---

## 🔮 5. Future Roadmap & Task List

- [ ] **Database Integration**: Migrate from in-memory session history to **SQLite** or **Supabase**.
- [ ] **Multi-Page PDFs**: Automate page-switching when the reader reaches the end of the current page.
- [ ] **Voice Cloning**: Implement a frontend for fine-tuning or uploading new `.pt` voice profiles.
- [ ] **Mobile Optimization**: Refine the sidebar and horizontal scroll for tablet/mobile viewports.

---

## 🚩 6. Common Gotchas
- **CORS**: Ensure `localhost` access is used for Web Speech API (STT) security requirements.
- **Port 8000**: Uvicorn might hang on exit; use `pkill -f uvicorn` if the address already in use.
- **Audio Blobs**: The Reader converts Base64 chunks to Blobs/ObjectURLs; ensure `stopReader()` is called to prevent memory leaks in long sessions.

---
**Lead Developer Note**: Keep the design aesthetic "Premium Dark Mode" (#0f172a / #f97316). Maintain 100% original codebase integrity with no external attribution.
