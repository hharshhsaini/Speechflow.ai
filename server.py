import os
import base64
import asyncio
import json
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from agent.tts import get_all_voices, synthesize, get_pipeline, resolve_voice_path
from agent.stt import transcribe, listen_and_transcribe
from pydub import AudioSegment
import random

load_dotenv()
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/voices")
async def voices():
    return get_all_voices()

class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0

@app.post("/synthesize")
async def api_synthesize(req: SynthesizeRequest):
    # Run TTS in a thread to not block
    audio_bytes = await asyncio.to_thread(synthesize, req.text, req.voice, req.speed)
    return Response(content=audio_bytes, media_type="audio/wav")

@app.post("/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    # Convert uploaded file to 16kHz mono PCM
    try:
        audio_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_io)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        pcm_bytes = audio.raw_data
        
        transcript = await asyncio.to_thread(transcribe, pcm_bytes)
        return {"transcript": transcript}
    except Exception as e:
        return {"transcript": f"[Error decoding audio: {str(e)}]"}

def convert_to_pcm(audio_bytes: bytes) -> bytes:
    try:
        audio_io = io.BytesIO(audio_bytes)
        audio = AudioSegment.from_file(audio_io)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        return audio.raw_data
    except Exception as e:
        print(f"Error converting audio to PCM: {e}")
        return b""

# Read presets from demo files

def get_preset(filename):
    path = os.path.join("demo", filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return ""

def get_random_quote():
    path = os.path.join("demo", "en.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
            return random.choice(lines) if lines else "No quotes available."
    return "Random quote file not found."

@app.get("/random-quote")
async def api_random_quote():
    return {"text": get_random_quote()}

@app.get("/gatsby")
async def api_gatsby():
    return {"text": get_preset("gatsby5k.md")}

@app.get("/frankenstein")
async def api_frankenstein():
    return {"text": get_preset("frankenstein5k.md")}

class TokenizeRequest(BaseModel):
    text: str
    voice: str = "af_heart"

@app.post("/tokenize")
async def api_tokenize(req: TokenizeRequest):
    prefix = req.voice[0].lower()
    from agent.tts import VOICE_LANG_MAP
    lang_code = VOICE_LANG_MAP.get(prefix, 'a')
    pipeline = get_pipeline(lang_code)
    
    # Simple tokenization: just run the generator and collect phonemes
    phonemes = []
    actual_voice = resolve_voice_path(req.voice)
    for _, ps, _ in pipeline(req.text, voice=actual_voice):
        phonemes.append(ps)
    return {"phonemes": " ".join(phonemes)}

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text_data = await websocket.receive_text()
            data = json.loads(text_data)
            
            text = data.get("text", "")
            voice = data.get("voice", "af_heart")
            speed = float(data.get("speed", 1.0))
            
            if not text:
                continue
                
            prefix = voice[0].lower()
            from agent.tts import VOICE_LANG_MAP
            lang_code = VOICE_LANG_MAP.get(prefix, 'a')
            pipeline = get_pipeline(lang_code)
            
            actual_voice = resolve_voice_path(voice)
            # Use the pipeline generator for streaming
            # Note: pipeline yields (graphemes, phonemes, audio)
            for _, ps, audio in pipeline(text, voice=actual_voice, speed=speed):
                if audio is not None:
                    # Convert numpy audio to bytes
                    buffer = io.BytesIO()
                    # sf.write expects (file, data, samplerate)
                    import soundfile as sf
                    sf.write(buffer, audio.numpy(), 24000, format='WAV')
                    wav_bytes = buffer.getvalue()
                    
                    # Send chunk
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": base64.b64encode(wav_bytes).decode("utf-8"),
                        "phonemes": ps
                    })
            
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Streaming error: {e}")

@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    processed_bytes = 0
    
    try:
        while True:
            # Receive binary chunk (PCM 16k mono)
            # Use wait_for to handle potential stalls
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
                
            if not data:
                break
                
            audio_buffer.extend(data)
            
            # Every ~8000 bytes (~250ms) of NEW data
            # Also don't transcribe if buffer is too long (> 30s) to keep it fast
            if len(audio_buffer) - processed_bytes >= 8000:
                # Only transcribe the last 15 seconds to keep it snappy
                MAX_BUF = 16000 * 2 * 15 # 15 seconds
                calc_buf = audio_buffer[-MAX_BUF:] if len(audio_buffer) > MAX_BUF else audio_buffer
                
                transcript = await asyncio.to_thread(transcribe, bytes(calc_buf))
                await websocket.send_json({"text": transcript, "is_final": False})
                processed_bytes = len(audio_buffer)
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"STT WebSocket error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pipeline = VoicePipeline()
    
    async def on_transcript(text):
        await websocket.send_json({"type": "transcript", "text": text})
        
    async def on_response(text):
        await websocket.send_json({"type": "response", "text": text})
        
    async def on_audio(audio_bytes):
        base64_audio = base64.b64encode(audio_bytes).decode("utf-8")
        await websocket.send_json({"type": "audio", "data": base64_audio})

    try:
        while True:
            text_data = await websocket.receive_text()
            data = json.loads(text_data)
            msg_type = data.get("type")
            
            if msg_type == "audio":
                # Decode base64 audio
                b64_data = data.get("data", "")
                if "," in b64_data:
                    b64_data = b64_data.split(",")[1]
                
                audio_bytes = base64.b64decode(b64_data)
                pcm_bytes = convert_to_pcm(audio_bytes)
                
                if pcm_bytes:
                    await pipeline.process_turn(pcm_bytes, on_transcript, on_response, on_audio)
                    
            elif msg_type == "set_voice":
                voice = data.get("voice", "af_heart")
                pipeline.set_voice(voice)
                
            elif msg_type == "reset":
                pipeline.reset()
                
            elif msg_type == "synthesize":
                # Direct synthesize request via WS
                text = data.get("text", "")
                if text:
                    await on_transcript(text) # Echo user text
                    
                    # Synthesize right away without LLM
                    audio_bytes = await asyncio.to_thread(synthesize, text, pipeline.current_voice, pipeline.speed)
                    await on_response(text) # In a pure TTS call, response is same as input
                    await on_audio(audio_bytes)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
