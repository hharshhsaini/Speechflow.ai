# SpeechFlow.ai | Pro Neural Studio

SpeechFlow.ai is a high-performance, multimodal AI Voice Agent suite built for ultra-accurate live transcription and professional-grade speech synthesis. It combines state-of-the-art neural models (Whisper + SpeechFlow/Kokoro) into a seamless, local-first studio experience.

## ✨ Features

- **Instant Live Transcription:** Neural typing directly into the workspace with ultra-high accuracy using Whisper `small` models.
- **Pro Synthesis Engine:** 20+ high-fidelity neural voices with full phoneme tracking and intonation control.
- **Multimodal Studio:** Integrated workspace for content creation, live streaming, and acoustic mastering.
- **Local-First Architecture:** All processing happens on-device for maximum privacy and zero latency.
- **Professional UI:** High-end glassmorphism design with real-time neural visualizers.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- `espeak-ng` (for phonemization)
  - **Mac:** `brew install espeak-ng`
  - **Ubuntu:** `sudo apt-get install espeak-ng`
  - **Windows:** Download from GitHub releases.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hharshhsaini/Speechflow.ai.git
   cd Speechflow.ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements_agent.txt
   ```

3. Setup environment:
   Create a `.env` file with your preferences (see `.env.example`).

### Running the Studio

Start the local AI engine:
```bash
python server.py
```
Then navigate to `http://localhost:8000` in your browser.

## 🛠️ Technology Stack

- **Backend:** FastAPI, WebSockets
- **STT Engine:** OpenAI Whisper (small)
- **TTS Engine:** SpeechFlow-82M / Kokoro
- **Frontend:** Vanilla JS, HTML5, Modern CSS (Glassmorphism)
- **Runtime:** UV / Python 3.12

## 📄 License

MIT License - Copyright (c) 2026

---
*Built with ❤️ for the future of voice interaction.*
