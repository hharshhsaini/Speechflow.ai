# SpeechFlowVoiceAgent — Live AI Voice Conversation

An end-to-end Live AI Voice Agent built on top of the `speechflow` TTS library. This project allows you to have natural, spoken conversations with an AI assistant in your browser.

## Features
- **Speech-to-Text (STT):** Uses OpenAI Whisper and WebRTC VAD to accurately detect when you start and stop speaking.
- **LLM Engine:** Powered by Anthropic's Claude 3.5 Sonnet to generate concise, engaging responses.
- **Text-to-Speech (TTS):** Uses local SpeechFlow models to synthesize high-quality, ultra-realistic audio.
- **Web Interface:** A sleek, dark-themed responsive web UI with WebSockets for real-time bidirectional streaming.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI, Python WebSockets |
| STT | OpenAI Whisper (`base` model), WebRTC VAD |
| LLM | Anthropic Claude (`claude-3-5-sonnet-20241022`) |
| TTS | SpeechFlow (Local weights) |
| Frontend | HTML5, JS (MediaRecorder API), CSS3 |

## Architecture

```text
       Browser                 FastAPI Server
+------------------+         +-------------------------------+
|                  |   ws    |                               |
|  Mic Recording   | ------> |  WebRTC VAD + Whisper (STT)   |
|   (WebM/Opus)    |         |               |               |
|                  |         |               v               |
|                  |         |     Claude 3.5 Sonnet (LLM)   |
|                  |         |               |               |
|  Audio Playback  | <------ |               v               |
|  (base64 WAV)    |   ws    |     SpeechFlow (TTS)          |
|                  |         |                               |
+------------------+         +-------------------------------+
```

## Setup Instructions

### 1. Prerequisites
You will need to install `espeak-ng` on your system for SpeechFlow's fallback text processing:
- **Ubuntu/Debian:** `sudo apt-get install espeak-ng`
- **Mac:** `brew install espeak`
- **Windows:** Download espeak-ng installer from GitHub

### 2. API Key
Get an Anthropic API Key from [console.anthropic.com](https://console.anthropic.com/).
Add it to your `.env` file:
```env
ANTHROPIC_API_KEY=sk-ant-api03...
```

### 3. Run the Agent
Use the provided runner scripts to automatically install dependencies and start the server:

**Mac/Linux:**
```bash
bash run.sh
```

**Windows:**
```cmd
run.bat
```

Alternatively, run manually:
```bash
pip install -r requirements_agent.txt
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

Open your browser to [http://localhost:8000](http://localhost:8000).

## Available Voices

- **American English:** Heart (F), Bella (F), Sarah (F), Sky (F), Nicole (F), Jessica (F), Kore (F), Nova (F), River (F), Alloy (F), Aoede (F), Adam (M), Michael (M)
- **British English:** Emma (F), Isabella (F), George (M), Lewis (M)

## Known Limitations
- The Whisper model runs locally on CPU by default. Depending on your machine, STT might take a couple of seconds.
- Currently uses the `base` Whisper model for speed; accuracy may vary for heavy accents.
- Requires microphone permissions in the browser to function.

## Screenshot
![Screenshot placeholder](https://via.placeholder.com/800x400?text=Voice+Agent+Screenshot)
