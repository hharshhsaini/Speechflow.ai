import os
import io
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
from agent.tts import synthesize, get_all_voices, validate_voice
from agent.stt import transcribe

app = FastAPI(title="SpeechFlow AI Agent")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class TokenizeRequest(BaseModel):
    text: str
    voice: str

class SynthesisRequest(BaseModel):
    text: str
    voice: str
    speed: Optional[float] = 1.0

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.get("/voices")
async def api_voices():
    registry = get_all_voices()
    # Group by language
    grouped = {}
    for vid, info in registry.items():
        lang = info['lang']
        if lang not in grouped:
            grouped[lang] = []
        # Add ID to info for frontend
        item = info.copy()
        item['id'] = vid
        grouped[lang].append(item)
    return grouped

@app.get("/voices/test/{voice_id}")
async def test_voice_endpoint(voice_id: str):
    works = validate_voice(voice_id)
    return {
        "voice": voice_id,
        "works": works,
        "error": None if works else "Synthesis failed or timed out"
    }

@app.post("/synthesize")
async def api_synthesize(req: SynthesisRequest):
    audio_bytes = synthesize(req.text, voice=req.voice, speed=req.speed)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Synthesis failed")
    return HTMLResponse(content=audio_bytes, media_type="audio/wav")

@app.post("/tokenize")
async def api_tokenize(req: TokenizeRequest):
    # This is a bit redundant now but kept for UI compatibility
    # We just return a placeholder or real tokens if needed
    return {"phonemes": "Neural tokens processed successfully."}

@app.post("/refine-transcript")
async def refine_transcript(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"refined": "", "changed": False}
    
    # In a real scenario, we might use Whisper to verify text
    # For now, we clean and normalize
    refined = text.strip()
    if refined and not refined.endswith(('.', '?', '!')):
        refined += '.'
        
    return {
        "refined": refined,
        "changed": refined != text
    }

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text_data = await websocket.receive_text()
            data = json.loads(text_data)
            text = data.get("text", "")
            voice = data.get("voice", "af_heart")
            speed = data.get("speed", 1.0)
            
            # Streaming synthesis
            # For simplicity, we use the shared SHARED_MODEL but in a streaming way
            # In KPipeline, the generator already yields chunks
            from agent.tts import get_pipeline, resolve_voice_path, VOICE_REGISTRY
            
            prefix = voice[0].lower()
            lang_code = VOICE_REGISTRY.get(voice, {}).get('lang_code', 'a')
            pipeline = get_pipeline(lang_code)
            actual_voice = resolve_voice_path(voice)
            
            for _, ps, audio in pipeline(text, voice=actual_voice, speed=speed):
                if audio is not None:
                    buffer = io.BytesIO()
                    import soundfile as sf
                    sf.write(buffer, audio.numpy(), 24000, format='WAV')
                    wav_bytes = buffer.getvalue()
                    
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": base64.b64encode(wav_bytes).decode("utf-8"),
                        "phonemes": ps
                    })
            
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass

@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    processed_bytes = 0
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            if not data:
                break
            audio_buffer.extend(data)
            if len(audio_buffer) - processed_bytes >= 8000:
                MAX_BUF = 16000 * 2 * 15 # 15s
                calc_buf = audio_buffer[-MAX_BUF:] if len(audio_buffer) > MAX_BUF else audio_buffer
                transcript = await asyncio.to_thread(transcribe, bytes(calc_buf))
                await websocket.send_json({"text": transcript, "is_final": False})
                processed_bytes = len(audio_buffer)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"STT Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
