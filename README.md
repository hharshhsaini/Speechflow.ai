# 🚀 SpeechFlow.ai Pro Neural Studio

**SpeechFlow.ai** is a professional-grade, local-first multimodal AI platform for high-fidelity voice synthesis, real-time transcription, and advanced document processing. Built with a focus on "Neural Precision," it provides a premium experience for creators and researchers.

![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)
![Python: 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![Engine: Kokoro](https://img.shields.io/badge/Engine-Kokoro-f97316.svg)

---

## ✨ Key Features

### 🎙️ Neural Studio
A minimalist workspace for high-fidelity voice synthesis.
- **Local Dictation**: Real-time STT using local Whisper models.
- **54+ Voice Profiles**: Curated neural voices across 5+ languages.
- **Neural Tokens**: Live visualization of phoneme-level generation.

### 📄 Document Reader (Pro)
Transform PDFs into immersive audiobooks with pixel-perfect synchronization.
- **Sentence-Level Chunking**: Natural pauses and fluid reading.
- **Smart Highlighting**: Hardware-locked word highlighting that never drifts.
- **Pause/Resume**: Full session persistence for long-form documents.

### 🧪 Voice Lab
A diagnostic suite for auditing the health of your neural library.
- **Mass Audit**: Automatically benchmark latency and quality across all 54 voices.
- **Neural Health Checks**: Verify synthesis integrity in real-time.

---

## 🛠️ Tech Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.12)
- **Frontend**: Premium Vanilla HTML5/CSS3 (Glassmorphism + Dark Obsidian)
- **TTS Engine**: [Kokoro](https://github.com/hexgrad/kokoro) (Local Neural Weights)
- **STT Engine**: [OpenAI Whisper](https://github.com/openai/whisper)
- **PDF Engine**: [PyMuPDF](https://github.com/pymupdf/PyMuPDF)

---

## 🚀 Quick Start (Local)

### 1. Clone & Install
```bash
git clone https://github.com/hharshhsaini/Speechflow.ai.git
cd Speechflow.ai
pip install -r requirements.txt
```

### 2. Setup Models
Ensure `local_models/kokoro-v1_0.pth` and `local_models/config.json` are present in the root directory.

### 3. Run Studio
```bash
./run.sh
```
Open **http://localhost:8000** in your browser.

---

## ☁️ Deployment (Render)

SpeechFlow is optimized for Docker-based deployment on **Render**.

### 1. GitHub Connection
Connect your repository to Render.

### 2. Web Service Configuration
- **Runtime**: `Docker`
- **Plan**: `Starter` (2GB RAM Required)
- **Port**: `8000`

### 3. Environment Variables
Ensure you add any necessary keys to the Render Environment panel.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created by the SpeechFlow Neural Engineering Team.*
