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
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response
from pydantic import BaseModel
from typing import Optional, List
from agent.tts import synthesize, tokenize_for_voice, get_all_voices, validate_voice, get_pipeline, resolve_voice_path, VOICE_REGISTRY
from agent.stt import transcribe

app = FastAPI(title="SpeechFlow AI Agent")

# Session storage for PDF data
PDF_SESSION = {
    "doc": None,
    "pages": [],
    "path": "",
    "version": 0,
    "page_image_cache": {},
    "page_image_tasks": {}
}

PDF_PAGE_IMAGE_DPI = 96

app.mount("/static", StaticFiles(directory="static"), name="static")

class TokenizeRequest(BaseModel):
    text: str
    voice: str
    translate: Optional[bool] = True

class SynthesisRequest(BaseModel):
    text: str
    voice: str
    speed: Optional[float] = 1.0
    translate: Optional[bool] = True

def build_pdf_page_info(page_num: int):
    if not PDF_SESSION["doc"] or page_num >= len(PDF_SESSION["doc"]):
        raise HTTPException(status_code=404, detail="Page not found")

    page = PDF_SESSION["doc"][page_num]
    text = page.get_text("text", sort=True)
    words_raw = page.get_text("words", sort=True)
    words = []
    for index, w in enumerate(words_raw):
        words.append({
            "x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3],
            "text": w[4], "page_num": page_num, "word_index": index
        })

    return {
        "page_num": page_num,
        "text": text,
        "word_count": len(words),
        "words": words,
        "width": page.rect.width,
        "height": page.rect.height
    }

def get_cached_pdf_page_info(page_num: int):
    if not PDF_SESSION["doc"] or page_num >= len(PDF_SESSION["doc"]):
        raise HTTPException(status_code=404, detail="Page not found")

    if page_num < len(PDF_SESSION["pages"]) and PDF_SESSION["pages"][page_num]:
        return PDF_SESSION["pages"][page_num]

    page_info = build_pdf_page_info(page_num)
    while len(PDF_SESSION["pages"]) <= page_num:
        PDF_SESSION["pages"].append(None)
    PDF_SESSION["pages"][page_num] = page_info
    return page_info

def render_pdf_page_image_bytes(page_num: int, dpi: int = PDF_PAGE_IMAGE_DPI):
    if not PDF_SESSION["doc"] or page_num >= len(PDF_SESSION["doc"]):
        raise HTTPException(status_code=404, detail="Page not found")
    page = PDF_SESSION["doc"][page_num]
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("png")

async def get_cached_pdf_page_image(page_num: int):
    if page_num in PDF_SESSION["page_image_cache"]:
        return PDF_SESSION["page_image_cache"][page_num]

    existing_task = PDF_SESSION["page_image_tasks"].get(page_num)
    if existing_task:
        return await existing_task

    async def _render():
        return await asyncio.to_thread(render_pdf_page_image_bytes, page_num)

    task = asyncio.create_task(_render())
    PDF_SESSION["page_image_tasks"][page_num] = task
    try:
        image_bytes = await task
        PDF_SESSION["page_image_cache"][page_num] = image_bytes
        return image_bytes
    finally:
        PDF_SESSION["page_image_tasks"].pop(page_num, None)

async def prewarm_pdf_page_images(page_numbers: list[int], session_version: int):
    for page_num in page_numbers:
        if PDF_SESSION["version"] != session_version:
            return
        if not PDF_SESSION["doc"] or page_num < 0 or page_num >= len(PDF_SESSION["doc"]):
            continue
        try:
            await get_cached_pdf_page_image(page_num)
        except Exception:
            continue

def get_reader_prewarm_pages(page_num: int):
    return [page_num, page_num + 1, page_num + 2, page_num - 1]

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.get("/health")
async def healthcheck():
    return {
        "status": "ok",
        "pdf_loaded": PDF_SESSION["doc"] is not None,
        "cached_pages": len(PDF_SESSION["page_image_cache"])
    }

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
        if PDF_SESSION["doc"] is not None:
            try:
                PDF_SESSION["doc"].close()
            except Exception:
                pass
        doc = fitz.open(tmp_path)
        PDF_SESSION["doc"] = doc
        PDF_SESSION["path"] = tmp_path
        PDF_SESSION["pages"] = []
        PDF_SESSION["version"] += 1
        PDF_SESSION["page_image_cache"] = {}
        PDF_SESSION["page_image_tasks"] = {}
        session_version = PDF_SESSION["version"]

        pages_data = []
        for i in range(len(doc)):
            page_info = build_pdf_page_info(i)
            PDF_SESSION["pages"].append(page_info)
            pages_data.append({
                "page_num": i,
                "text": page_info["text"],
                "word_count": page_info["word_count"]
            })

        asyncio.create_task(prewarm_pdf_page_images([0, 1, 2, 3], session_version))
        return {"total_pages": len(doc), "pages": pages_data, "session_version": session_version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF error: {e}")

@app.get("/pdf-page-image/{page_num}")
async def get_pdf_page_image(page_num: int):
    if not PDF_SESSION["doc"]:
        raise HTTPException(status_code=404, detail="No PDF loaded")
    try:
        img_data = await get_cached_pdf_page_image(page_num)
        return Response(
            content=img_data,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pdf-page-data/{page_num}")
async def get_pdf_page_data(page_num: int):
    page_data = get_cached_pdf_page_info(page_num)
    asyncio.create_task(prewarm_pdf_page_images(get_reader_prewarm_pages(page_num), PDF_SESSION["version"]))
    return page_data

def split_into_sentences(words):
    """Group words into sentences based on punctuation."""
    sentences = []
    current_sent = []
    for w in words:
        current_sent.append(w)
        # End sentence on common sentence-final punctuation across supported scripts.
        if re.search(r'[.!?。！？]$', w["text"]):
            sentences.append(current_sent)
            current_sent = []
    if current_sent:
        sentences.append(current_sent)
    return sentences

def split_reader_chunks(words, max_chars: int = 120):
    chunks = []
    current = []
    current_chars = 0

    for word in words:
        text = word["text"]
        current.append(word)
        current_chars += len(text) + (1 if len(current) > 1 else 0)

        is_strong_break = bool(re.search(r'[.!?。！？]$', text))
        is_soft_break = bool(re.search(r'[,;:،，；：]$', text))
        if is_strong_break or current_chars >= max_chars or (is_soft_break and current_chars >= max_chars * 0.65):
            chunks.append(current)
            current = []
            current_chars = 0

    if current:
        chunks.append(current)

    return chunks

def join_reader_words(words):
    return " ".join(word["text"] for word in words).strip()

def build_reader_timing(words):
    weighted_words = []
    total_weight = 0.0

    for word in words:
        text = word["text"].strip()
        core = re.sub(r'[\W_]+', '', text, flags=re.UNICODE)
        weight = float(max(1, len(core) or len(text)))
        if re.search(r'[,;:،，；：]$', text):
            weight += 0.75
        if re.search(r'[.!?。！？]$', text):
            weight += 1.35
        weighted_words.append((word, weight))
        total_weight += weight

    total_weight = total_weight or float(len(weighted_words) or 1)
    current = 0.0
    timing = []

    for word, weight in weighted_words:
        start_ratio = current / total_weight
        current += weight
        end_ratio = current / total_weight
        timing.append({
            "word_index": word["word_index"],
            "start_ratio": start_ratio,
            "end_ratio": end_ratio,
            "word": word["text"]
        })

    return timing

def consume_result_words(words, graphemes: str, is_last: bool = False):
    if not words:
        return [], 0
    if is_last:
        return words, len(words)

    target = re.sub(r'\s+', ' ', graphemes or '').strip()
    if not target:
        return [], 0

    target_len = len(target)
    consumed_words = []
    built = ""

    for idx, word in enumerate(words):
        built = f"{built} {word['text']}".strip() if built else word["text"]
        consumed_words.append(word)
        if len(built) >= target_len * 0.9:
            return consumed_words, idx + 1

    return words, len(words)

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

            page_data = get_cached_pdf_page_info(page_num)
            words_to_read = page_data["words"][start_word_idx:]
            sentences = split_into_sentences(words_to_read)
            
            for sent_words in sentences:
                reader_chunks = split_reader_chunks(sent_words)
                lang_code = VOICE_REGISTRY.get(voice, {}).get('lang_code', 'a')
                pipeline = get_pipeline(lang_code)
                actual_voice = resolve_voice_path(voice)

                for chunk_words in reader_chunks:
                    text_to_synthesize = join_reader_words(chunk_words)
                    results = [result for result in pipeline(text_to_synthesize, voice=actual_voice, speed=speed) if result.audio is not None]
                    remaining_words = chunk_words[:]

                    for result_index, result in enumerate(results):
                        result_words, consumed = consume_result_words(
                            remaining_words,
                            result.graphemes,
                            is_last=result_index == len(results) - 1
                        )
                        if consumed:
                            remaining_words = remaining_words[consumed:]
                        timing_data = build_reader_timing(result_words or chunk_words)

                        import soundfile as sf
                        buffer = io.BytesIO()
                        sf.write(buffer, result.audio.numpy(), 24000, format='WAV')

                        await websocket.send_json({
                            "type": "audio_with_timing",
                            "audio": base64.b64encode(buffer.getvalue()).decode("utf-8"),
                            "timing": timing_data,
                            "text": result.graphemes or text_to_synthesize
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
    last_transcript = ""
    language = None
    STT_BYTES_PER_SECOND = 16000 * 2
    STT_INTERIM_MIN_BYTES = STT_BYTES_PER_SECOND
    STT_INTERIM_STEP_BYTES = STT_BYTES_PER_SECOND // 2
    STT_MAX_BUF = STT_BYTES_PER_SECOND * 20
    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            if message["type"] == "websocket.disconnect":
                break

            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                    language = payload.get("lang") or None
                except json.JSONDecodeError:
                    pass
                continue

            data = message.get("bytes")
            if data is None:
                continue

            if not data:
                if len(audio_buffer) > processed_bytes:
                    calc_buf = audio_buffer[-STT_MAX_BUF:] if len(audio_buffer) > STT_MAX_BUF else audio_buffer
                    transcript = await asyncio.to_thread(transcribe, bytes(calc_buf), language)
                    await websocket.send_json({"text": transcript, "is_final": True})
                break
            audio_buffer.extend(data)

            if len(audio_buffer) < STT_INTERIM_MIN_BYTES:
                continue

            if len(audio_buffer) - processed_bytes >= STT_INTERIM_STEP_BYTES:
                calc_buf = audio_buffer[-STT_MAX_BUF:] if len(audio_buffer) > STT_MAX_BUF else audio_buffer
                transcript = await asyncio.to_thread(transcribe, bytes(calc_buf), language)
                if transcript != last_transcript:
                    await websocket.send_json({"text": transcript, "is_final": False})
                    last_transcript = transcript
                processed_bytes = len(audio_buffer)
    except WebSocketDisconnect:
        pass

@app.post("/synthesize")
async def api_synthesize(req: SynthesisRequest):
    audio_bytes = synthesize(req.text, voice=req.voice, speed=req.speed, translate=req.translate)
    if not audio_bytes:
        raise HTTPException(status_code=500, detail="Synthesis failed")
    return HTMLResponse(content=audio_bytes, media_type="audio/wav")

@app.post("/tokenize")
async def api_tokenize(req: TokenizeRequest):
    token_data = await asyncio.to_thread(tokenize_for_voice, req.text, req.voice, req.translate)
    if not token_data.get("phonemes"):
        raise HTTPException(status_code=500, detail="Tokenization failed")
    return token_data

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
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000"))
    )
