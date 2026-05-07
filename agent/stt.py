import os
import io
import wave
import torch
import numpy as np
try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None  # Not available in Docker/server environments
try:
    import webrtcvad
except (ImportError, OSError):
    webrtcvad = None
import whisper
from dotenv import load_dotenv

load_dotenv()

_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_name = os.getenv("WHISPER_MODEL", "small")
        print(f"Loading Whisper model: {model_name}...")
        _whisper_model = whisper.load_model(model_name)
        print("Whisper model loaded successfully.")
    return _whisper_model

def record_until_silence(sample_rate=16000, aggressiveness=2, silence_chunks=30):
    vad = webrtcvad.Vad(aggressiveness)
    frame_duration_ms = 30
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
    
    recorded_frames = []
    has_speech_started = False
    silent_chunk_count = 0

    with sd.RawInputStream(samplerate=sample_rate, blocksize=frame_size,
                           dtype='int16', channels=1) as stream:
        while True:
            frame, overflowed = stream.read(frame_size)
            frame_bytes = bytes(frame)
            is_speech = vad.is_speech(frame_bytes, sample_rate)
            
            if not has_speech_started:
                if is_speech:
                    has_speech_started = True
                    recorded_frames.append(frame_bytes)
            else:
                recorded_frames.append(frame_bytes)
                if not is_speech:
                    silent_chunk_count += 1
                else:
                    silent_chunk_count = 0
                if silent_chunk_count >= silence_chunks:
                    break
    return b"".join(recorded_frames)

def transcribe(audio_bytes: bytes, language: str | None = None) -> str:
    if not audio_bytes:
        return ""

    model = get_whisper_model()
    audio_int16 = np.frombuffer(audio_bytes, np.int16)
    if audio_int16.size == 0:
        return ""

    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    audio_float32 = audio_float32 - np.mean(audio_float32)
    peak = np.max(np.abs(audio_float32))
    if peak > 0:
        audio_float32 = audio_float32 / peak
    
    result = model.transcribe(
        audio_float32, 
        fp16=torch.cuda.is_available(),
        language=language,
        task="transcribe",
        beam_size=3,
        best_of=3,
        patience=1.2,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.15,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        verbose=False
    )
    return result["text"].strip()

def listen_and_transcribe() -> str:
    audio_bytes = record_until_silence()
    if len(audio_bytes) > 0:
        return transcribe(audio_bytes)
    return ""

if __name__ == "__main__":
    text = listen_and_transcribe()
    print(f"Transcription: {text}")
