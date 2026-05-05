from speechflow import KPipeline, KModel
import numpy as np
import io
import soundfile as sf
import os

# Create a single local model instance shared across pipelines
# We use the local_models path because the Hugging Face repo has been renamed.
# Note: Ensure local_models/config.json and kokoro-v1_0.pth exist.
LOCAL_MODEL_PATH = "local_models/kokoro-v1_0.pth"
LOCAL_CONFIG_PATH = "local_models/config.json"
LOCAL_VOICES_DIR = "local_models/voices"

if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_CONFIG_PATH):
    SHARED_MODEL = KModel(config=LOCAL_CONFIG_PATH, model=LOCAL_MODEL_PATH).eval()
else:
    print("Warning: local_models not found. Defaulting to Hugging Face download (which may fail due to repo rename).")
    SHARED_MODEL = KModel().eval()

# Cache for pipelines per lang_code
_pipelines = {}

VOICE_LANG_MAP = {
    'a': 'a', # American English
    'b': 'b', # British English
    'j': 'j', # Japanese
    'z': 'z', # Chinese
    'e': 'e', # Spanish
    'f': 'f', # French
    'h': 'h', # Hindi
    'k': 'k', # Korean
    'p': 'p', # Portuguese
    'i': 'i'  # Italian
}

def resolve_voice_path(voice: str) -> str:
    # Check if voice exists locally, else pass string and rely on HF download (which might fail)
    voice_path = os.path.join(LOCAL_VOICES_DIR, f"{voice}.pt")
    if os.path.exists(voice_path):
        return voice_path
    return voice

def get_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in _pipelines:
        _pipelines[lang_code] = KPipeline(lang_code=lang_code, model=SHARED_MODEL)
    return _pipelines[lang_code]

def synthesize(text: str, voice: str = 'af_heart', speed: float = 1.0) -> bytes:
    # Determine lang_code from first character
    prefix = voice[0].lower()
    lang_code = VOICE_LANG_MAP.get(prefix, 'a')
    
    pipeline = get_pipeline(lang_code)
    
    actual_voice = resolve_voice_path(voice)

    generator = pipeline(text, voice=actual_voice, speed=speed)
    
    audio_chunks = []
    for _, _, audio in generator:
        if audio is not None:
            audio_chunks.append(audio)
            
    if not audio_chunks:
        return b""
        
    full_audio = np.concatenate(audio_chunks)
    
    # Convert float32 numpy array to WAV bytes
    buffer = io.BytesIO()
    sf.write(buffer, full_audio, 24000, format='WAV')
    return buffer.getvalue()

def get_all_voices() -> list[dict]:
    # Hardcoded minimum requested voices
    voices = [
        {"id": "af_heart", "name": "Heart", "language": "American English", "gender": "Female"},
        {"id": "af_bella", "name": "Bella", "language": "American English", "gender": "Female"},
        {"id": "af_sarah", "name": "Sarah", "language": "American English", "gender": "Female"},
        {"id": "af_sky", "name": "Sky", "language": "American English", "gender": "Female"},
        {"id": "af_nicole", "name": "Nicole", "language": "American English", "gender": "Female"},
        {"id": "af_jessica", "name": "Jessica", "language": "American English", "gender": "Female"},
        {"id": "af_kore", "name": "Kore", "language": "American English", "gender": "Female"},
        {"id": "af_nova", "name": "Nova", "language": "American English", "gender": "Female"},
        {"id": "af_river", "name": "River", "language": "American English", "gender": "Female"},
        {"id": "af_alloy", "name": "Alloy", "language": "American English", "gender": "Female"},
        {"id": "af_aoede", "name": "Aoede", "language": "American English", "gender": "Female"},
        {"id": "am_adam", "name": "Adam", "language": "American English", "gender": "Male"},
        {"id": "am_michael", "name": "Michael", "language": "American English", "gender": "Male"},
        {"id": "bf_emma", "name": "Emma", "language": "British English", "gender": "Female"},
        {"id": "bf_isabella", "name": "Isabella", "language": "British English", "gender": "Female"},
        {"id": "bm_george", "name": "George", "language": "British English", "gender": "Male"},
        {"id": "bm_lewis", "name": "Lewis", "language": "British English", "gender": "Male"}
    ]
    return voices

if __name__ == "__main__":
    test_text = "This is a test of the Text to Speech agent module."
    print("Synthesizing test audio...")
    audio_bytes = synthesize(test_text, voice='af_heart')
    with open("test_output.wav", "wb") as f:
        f.write(audio_bytes)
    print("Saved to test_output.wav")
