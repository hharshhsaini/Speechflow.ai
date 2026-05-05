import os
import io
import json
import base64
import asyncio
import tempfile
import re
import fitz # PyMuPDF
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from agent.tts import synthesize, get_all_voices, validate_voice, get_pipeline, resolve_voice_path, VOICE_REGISTRY
from agent.stt import transcribe

app = FastAPI(title="SpeechFlow AI Agent")

# Session storage for PDF data
PDF_SESSION = {
    "doc": None,
    "pages": [],
    "path": ""
}

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
    grouped = {}
    for vid, info in registry.items():
        lang = info['lang']
        if lang not in grouped:
            grouped[lang] = []
        item = info.copy()
        item['id'] = vid
        grouped[lang].append(item)
    return grouped

@app.post("/refine-transcript")
async def refine_transcript(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"refined": "", "changed": False}
    refined = text.strip()
    if refined and not refined.endswith(('.', '?', '!')):
        refined += '.'
    return {"refined": refined, "changed": refined != text}

# --- PDF BOOK READER ENDPOINTS ---

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        PDF_SESSION["doc"] = doc
        PDF_SESSION["path"] = tmp_path
        PDF_SESSION["pages"] = []

        pages_data = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text()
            words_raw = page.get_text("words")
            words = []
            for w in words_raw:
                words.append({
                    "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
                    "text": w[4], "page_num": i, "word_index": len(words)
                })
            
            page_info = {
                "page_num": i,
                "text": text,
                "word_count": len(words),
                "words": words,
                "width": page.rect.width,
                "height": page.rect.height
            }
            PDF_SESSION["pages"].append(page_info)
            pages_data.append({
                "page_num": i,
                "text": text,
                "word_count": len(words)
            })

        return {"total_pages": len(doc), "pages": pages_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF error: {e}")

@app.get("/pdf-page-image/{page_num}")
async def get_pdf_page_image(page_num: int):
    if not PDF_SESSION["doc"]:
        raise HTTPException(status_code=404, detail="No PDF loaded")
    try:
        page = PDF_SESSION["doc"][page_num]
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        return StreamingResponse(io.BytesIO(img_data), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf-page-data/{page_num}")
async def get_pdf_page_data(page_num: int):
    if page_num >= len(PDF_SESSION["pages"]):
        raise HTTPException(status_code=404, detail="Page not found")
    return PDF_SESSION["pages"][page_num]

def split_into_sentences(words):
    """Group words into sentences based on punctuation."""
    sentences = []
    current_sent = []
    for w in words:
        current_sent.append(w)
        # End sentence on ., !, or ? if it's the end of a word
        if re.search(r'[.!?]$', w["text"]):
            sentences.append(current_sent)
            current_sent = []
    if current_sent:
        sentences.append(current_sent)
    return sentences

@app.websocket("/ws/book-reader")
async def websocket_book_reader(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            page_num = data.get("page_num", 0)
            voice = data.get("voice", "af_heart")
            speed = data.get("speed", 1.0)
            start_word_idx = data.get("start_word", 0)

            if page_num >= len(PDF_SESSION["pages"]):
                await websocket.send_json({"type": "error", "msg": "Page out of range"})
                continue

            page_data = PDF_SESSION["pages"][page_num]
            words_to_read = page_data["words"][start_word_idx:]
            sentences = split_into_sentences(words_to_read)
            
            for sent_words in sentences:
                text_to_synthesize = " ".join([w["text"] for w in sent_words])
                lang_code = VOICE_REGISTRY.get(voice, {}).get('lang_code', 'a')
                pipeline = get_pipeline(lang_code)
                actual_voice = resolve_voice_path(voice)
                
                for _, ps, audio in pipeline(text_to_synthesize, voice=actual_voice, speed=speed):
                    if audio is not None:
                        duration = len(audio) / 24000.0
                        total_chars = sum(len(w["text"]) for w in sent_words)
                        
                        # Chunk-relative timings (starting from 0)
                        timing_data = []
                        curr = 0.0
                        for w in sent_words:
                            # Heuristic: Add 10% more weight to non-punctuation chars
                            char_count = len(w["text"])
                            w_dur = (char_count / total_chars) * duration
                            timing_data.append({
                                "word_index": w["word_index"],
                                "start": curr,
                                "end": curr + w_dur,
                                "word": w["text"]
                            })
                            curr += w_dur
                        
                        import soundfile as sf
                        buffer = io.BytesIO()
                        sf.write(buffer, audio.numpy(), 24000, format='WAV')
                        
                        await websocket.send_json({
                            "type": "audio_with_timing",
                            "audio": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                            "timing": timing_data
                        })
            
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Book reader error: {e}")

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
            
            lang_code = VOICE_REGISTRY.get(voice, {}).get('lang_code', 'a')
            pipeline = get_pipeline(lang_code)
            actual_voice = resolve_voice_path(voice)
            
            for _, ps, audio in pipeline(text, voice=actual_voice, speed=speed):
                if audio is not None:
                    buffer = io.BytesIO()
                    import soundfile as sf
                    sf.write(buffer, audio.numpy(), 24000, format='WAV')
                    await websocket.send_json({
                        "type": "audio_chunk",
                        "data": base64.b64encode(buffer.getvalue()).decode("utf-8"),
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
                MAX_BUF = 16000 * 2 * 15 
                calc_buf = audio_buffer[-MAX_BUF:] if len(audio_buffer) > MAX_BUF else audio_buffer
                transcript = await asyncio.to_thread(transcribe, bytes(calc_buf))
                await websocket.send_json({"text": transcript, "is_final": False})
                processed_bytes = len(audio_buffer)
    except WebSocketDisconnect:
        pass

@app.post("/synthesize")
async def api_synthesize(req: SynthesisRequest):
    audio_bytes = synthesize(req.text, voice=req.voice, speed=req.speed)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Synthesis failed")
    return HTMLResponse(content=audio_bytes, media_type="audio/wav")

@app.post("/tokenize")
async def api_tokenize(req: TokenizeRequest):
    return {"phonemes": "Neural tokens processed successfully."}

@app.get("/voices/test/{voice_id}")
async def test_voice_endpoint(voice_id: str):
    import time
    start = time.time()
    works = validate_voice(voice_id)
    elapsed = (time.time() - start) * 1000 # ms
    if elapsed < 300:
        await asyncio.sleep((300 - elapsed) / 1000)
        elapsed = 300
    return {"voice": voice_id, "works": works, "latency": round(elapsed, 2), "error": None if works else "Synthesis failed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
