import os
import io
import numpy as np
import soundfile as sf
from speechflow import KPipeline, KModel

# Create a single local model instance shared across pipelines
LOCAL_MODEL_PATH = "local_models/kokoro-v1_0.pth"
LOCAL_CONFIG_PATH = "local_models/config.json"
LOCAL_VOICES_DIR = "local_models/voices"

if os.path.exists(LOCAL_MODEL_PATH) and os.path.exists(LOCAL_CONFIG_PATH):
    SHARED_MODEL = KModel(config=LOCAL_CONFIG_PATH, model=LOCAL_MODEL_PATH).eval()
else:
    print("Warning: local_models not found. Defaulting to Hugging Face download.")
    SHARED_MODEL = KModel().eval()

# Cache for pipelines per lang_code
_pipeline_cache = {}

VOICE_REGISTRY = {
  # American English (lang_code='a')
  'af_heart':    {'name': 'Heart',    'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'af_bella':    {'name': 'Bella',    'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'af_sarah':    {'name': 'Sarah',    'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'af_sky':      {'name': 'Sky',      'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'af_nicole':   {'name': 'Nicole',   'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'B'},
  'af_jessica':  {'name': 'Jessica',  'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'af_kore':     {'name': 'Kore',     'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'af_nova':     {'name': 'Nova',     'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'af_river':    {'name': 'River',    'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'af_alloy':    {'name': 'Alloy',    'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'B'},
  'af_aoede':    {'name': 'Aoede',    'lang': 'American English', 'gender': 'Female', 'lang_code': 'a', 'quality': 'A'},
  'am_adam':     {'name': 'Adam',     'lang': 'American English', 'gender': 'Male',   'lang_code': 'a', 'quality': 'B'},
  'am_michael':  {'name': 'Michael',  'lang': 'American English', 'gender': 'Male',   'lang_code': 'a', 'quality': 'A'},
  'am_echo':     {'name': 'Echo',     'lang': 'American English', 'gender': 'Male',   'lang_code': 'a', 'quality': 'A'},
  'am_eric':     {'name': 'Eric',     'lang': 'American English', 'gender': 'Male',   'lang_code': 'a', 'quality': 'A'},
  'am_fenrir':   {'name': 'Fenrir',   'lang': 'American English', 'gender': 'Male',   'lang_code': 'a', 'quality': 'A'},
  'am_liam':     {'name': 'Liam',     'lang': 'American English', 'gender': 'Male',   'lang_code': 'a', 'quality': 'A'},
  'am_onyx':     {'name': 'Onyx',     'lang': 'American English', 'gender': 'Male',   'lang_code': 'a', 'quality': 'A'},
  'am_puck':     {'name': 'Puck',     'lang': 'American English', 'gender': 'Male',   'lang_code': 'a', 'quality': 'A'},
  # British English (lang_code='b')
  'bf_emma':     {'name': 'Emma',     'lang': 'British English',  'gender': 'Female', 'lang_code': 'b', 'quality': 'A'},
  'bf_isabella': {'name': 'Isabella', 'lang': 'British English',  'gender': 'Female', 'lang_code': 'b', 'quality': 'A'},
  'bf_alice':    {'name': 'Alice',    'lang': 'British English',  'gender': 'Female', 'lang_code': 'b', 'quality': 'A'},
  'bf_lily':     {'name': 'Lily',     'lang': 'British English',  'gender': 'Female', 'lang_code': 'b', 'quality': 'A'},
  'bm_george':   {'name': 'George',   'lang': 'British English',  'gender': 'Male',   'lang_code': 'b', 'quality': 'A'},
  'bm_lewis':    {'name': 'Lewis',    'lang': 'British English',  'gender': 'Male',   'lang_code': 'b', 'quality': 'A'},
  'bm_daniel':   {'name': 'Daniel',   'lang': 'British English',  'gender': 'Male',   'lang_code': 'b', 'quality': 'A'},
  'bm_fable':    {'name': 'Fable',    'lang': 'British English',  'gender': 'Male',   'lang_code': 'b', 'quality': 'A'},
  # Japanese (lang_code='j')
  'jf_alpha':    {'name': 'Alpha',    'lang': 'Japanese',         'gender': 'Female', 'lang_code': 'j', 'quality': 'A'},
  'jf_gongitsune':{'name':'Gongitsune','lang':'Japanese',         'gender': 'Female', 'lang_code': 'j', 'quality': 'A'},
  'jm_kumo':     {'name': 'Kumo',     'lang': 'Japanese',         'gender': 'Male',   'lang_code': 'j', 'quality': 'A'},
  # Chinese Mandarin (lang_code='z')
  'zf_xiaobei':  {'name': 'Xiaobei',  'lang': 'Chinese',          'gender': 'Female', 'lang_code': 'z', 'quality': 'A'},
  'zm_yunjian':  {'name': 'Yunjian',  'lang': 'Chinese',          'gender': 'Male',   'lang_code': 'z', 'quality': 'A'},
  # Spanish (lang_code='e')
  'ef_dora':     {'name': 'Dora',     'lang': 'Spanish',          'gender': 'Female', 'lang_code': 'e', 'quality': 'A'},
  'em_alex':     {'name': 'Alex',     'lang': 'Spanish',          'gender': 'Male',   'lang_code': 'e', 'quality': 'A'},
  # French (lang_code='f')
  'ff_siwis':    {'name': 'Siwis',    'lang': 'French',           'gender': 'Female', 'lang_code': 'f', 'quality': 'A'},
  # Hindi (lang_code='h')
  'hf_alpha':    {'name': 'Alpha',    'lang': 'Hindi',            'gender': 'Female', 'lang_code': 'h', 'quality': 'A'},
  'hm_omega':    {'name': 'Omega',    'lang': 'Hindi',            'gender': 'Male',   'lang_code': 'h', 'quality': 'A'},
  # Italian (lang_code='i')
  'if_sara':     {'name': 'Sara',     'lang': 'Italian',          'gender': 'Female', 'lang_code': 'i', 'quality': 'A'},
  'im_nicola':   {'name': 'Nicola',   'lang': 'Italian',          'gender': 'Male',   'lang_code': 'i', 'quality': 'A'},
  # Korean (lang_code='k')
  'kf_alpha':    {'name': 'Alpha',    'lang': 'Korean',           'gender': 'Female', 'lang_code': 'k', 'quality': 'A'},
  # Portuguese (lang_code='p')
  'pf_dora':     {'name': 'Dora',     'lang': 'Portuguese',       'gender': 'Female', 'lang_code': 'p', 'quality': 'A'},
  'pm_alex':     {'name': 'Alex',     'lang': 'Portuguese',       'gender': 'Male',   'lang_code': 'p', 'quality': 'A'},
}

def resolve_voice_path(voice: str) -> str:
    voice_path = os.path.join(LOCAL_VOICES_DIR, f"{voice}.pt")
    if os.path.exists(voice_path):
        return voice_path
    return voice

def get_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in _pipeline_cache:
        print(f"Initializing pipeline for lang_code: {lang_code}")
        _pipeline_cache[lang_code] = KPipeline(lang_code=lang_code, model=SHARED_MODEL)
    return _pipeline_cache[lang_code]

def synthesize(text: str, voice: str = 'af_heart', speed: float = 1.0) -> bytes:
    try:
        # Determine lang_code
        if voice in VOICE_REGISTRY:
            lang_code = VOICE_REGISTRY[voice]['lang_code']
        else:
            lang_code = voice[0].lower() if voice else 'a'
            
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
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, 24000, format='WAV')
        return buffer.getvalue()
        
    except Exception as e:
        print(f"Synthesis error for voice {voice}: {e}")
        if voice != 'af_heart':
            return synthesize(text, 'af_heart', speed)
        return b""

def validate_voice(voice_id: str) -> bool:
    try:
        audio = synthesize("test", voice_id)
        return len(audio) > 0
    except:
        return False

def get_all_voices() -> dict:
    return VOICE_REGISTRY
