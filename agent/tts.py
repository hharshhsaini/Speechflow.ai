import os
import io
import json
import re
import numpy as np
import soundfile as sf
from functools import lru_cache
from urllib import parse, request
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
_quiet_pipeline_cache = {}

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

VOICE_ASSET_FALLBACKS = {
    # Kokoro does not currently ship a dedicated Korean pack in this repo/upstream voice set.
    # Reuse the closest matching "Alpha" speaker embedding so Korean synthesis still works.
    'kf_alpha': 'hf_alpha',
}

LANGUAGE_SETTINGS = {
    'a': {'translate_code': 'en', 'health_text': 'Neural integrity check.'},
    'b': {'translate_code': 'en', 'health_text': 'Neural integrity check.'},
    'e': {'translate_code': 'es', 'health_text': 'Comprobacion de integridad neural.'},
    'f': {'translate_code': 'fr', 'health_text': 'Verification de l integrite neuronale.'},
    'h': {'translate_code': 'hi', 'health_text': 'न्यूरल अखंडता जांच।'},
    'i': {'translate_code': 'it', 'health_text': 'Controllo di integrita neurale.'},
    'j': {'translate_code': 'ja', 'health_text': 'ニューラル整合性チェック。'},
    'k': {'translate_code': 'ko', 'health_text': '신경 무결성 점검입니다.'},
    'p': {'translate_code': 'pt', 'health_text': 'Verificacao de integridade neural.'},
    'z': {'translate_code': 'zh-CN', 'health_text': '神经完整性检查。'},
}

TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
TRANSLATE_FALLBACK_URL = "https://clients5.google.com/translate_a/t"
TRANSLATE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/x-www-form-urlencoded",
}
TRANSLATE_TIMEOUT = 20
TRANSLATE_CHUNK_LIMIT = 3500


def get_voice_lang_code(voice: str) -> str:
    if voice in VOICE_REGISTRY:
        return VOICE_REGISTRY[voice]['lang_code']
    return voice[0].lower() if voice else 'a'


def get_voice_settings(voice: str) -> dict:
    return LANGUAGE_SETTINGS.get(get_voice_lang_code(voice), LANGUAGE_SETTINGS['a'])


def _split_translation_text(text: str, limit: int = TRANSLATE_CHUNK_LIMIT) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]

    def flush_segment(chunks: list[str], current: list[str]) -> None:
        if current:
            chunks.append("\n\n".join(current).strip())
            current.clear()

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', normalized) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        segments = [paragraph]
        if len(paragraph) > limit:
            segments = [s.strip() for s in re.split(r'(?<=[.!?。！？])\s+', paragraph) if s.strip()]
            if not segments:
                segments = [paragraph]

        for segment in segments:
            if len(segment) > limit:
                words = segment.split()
                if len(words) > 1:
                    pieces: list[str] = []
                    piece = ""
                    for word in words:
                        trial = f"{piece} {word}".strip()
                        if len(trial) > limit and piece:
                            pieces.append(piece)
                            piece = word
                        else:
                            piece = trial
                    if piece:
                        pieces.append(piece)
                    subsegments = pieces
                else:
                    subsegments = [segment[i:i + limit] for i in range(0, len(segment), limit)]
            else:
                subsegments = [segment]

            for subsegment in subsegments:
                projected_len = current_len + len(subsegment) + (2 if current else 0)
                if current and projected_len > limit:
                    flush_segment(chunks, current)
                    current_len = 0
                current.append(subsegment)
                current_len += len(subsegment) + (2 if len(current) > 1 else 0)

    flush_segment(chunks, current)
    return chunks or [normalized]


@lru_cache(maxsize=256)
def _translate_chunk(text: str, target_lang: str) -> str:
    translated = ""
    attempts = [
        (
            TRANSLATE_URL,
            {
                "client": "gtx",
                "sl": "auto",
                "tl": target_lang,
                "dt": "t",
                "dj": "1",
            },
            parse.urlencode({"q": text}).encode("utf-8"),
        ),
        (
            TRANSLATE_FALLBACK_URL,
            {
                "client": "dict-chrome-ex",
                "sl": "auto",
                "tl": target_lang,
                "q": text,
            },
            None,
        ),
    ]

    last_error = None
    for base_url, params, payload in attempts:
        try:
            req = request.Request(
                f"{base_url}?{parse.urlencode(params)}",
                data=payload,
                headers=TRANSLATE_HEADERS,
            )
            with request.urlopen(req, timeout=TRANSLATE_TIMEOUT) as resp:
                data = json.load(resp)
            if isinstance(data, dict):
                translated = "".join(part.get("trans", "") for part in data.get("sentences", []))
            else:
                translated = "".join(part[0] for part in data if part and part[0])
            if translated.strip():
                break
        except Exception as e:
            last_error = e

    translated = translated.strip()
    if not translated:
        raise ValueError(f"Empty translation response: {last_error}")
    return translated


def translate_for_voice(text: str, voice: str) -> str:
    normalized = text.strip()
    if not normalized:
        return text

    settings = get_voice_settings(voice)
    target_lang = settings['translate_code']
    translated_chunks = []

    for chunk in _split_translation_text(normalized):
        try:
            translated_chunks.append(_translate_chunk(chunk, target_lang))
        except Exception as e:
            print(f"Translation error for voice {voice} ({target_lang}): {e}")
            translated_chunks.append(chunk)

    translated_text = "\n\n".join(part for part in translated_chunks if part).strip()
    return translated_text or normalized

def resolve_voice_path(voice: str) -> str:
    voice_path = os.path.join(LOCAL_VOICES_DIR, f"{voice}.pt")
    if os.path.exists(voice_path):
        return voice_path
    if voice in VOICE_ASSET_FALLBACKS:
        return VOICE_ASSET_FALLBACKS[voice]
    return voice

def get_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in _pipeline_cache:
        print(f"Initializing pipeline for lang_code: {lang_code}")
        _pipeline_cache[lang_code] = KPipeline(lang_code=lang_code, model=SHARED_MODEL)
    return _pipeline_cache[lang_code]

def get_quiet_pipeline(lang_code: str) -> KPipeline:
    if lang_code not in _quiet_pipeline_cache:
        _quiet_pipeline_cache[lang_code] = KPipeline(lang_code=lang_code, model=False)
    return _quiet_pipeline_cache[lang_code]

def prepare_text_for_voice(text: str, voice: str = 'af_heart', translate: bool = False) -> str:
    return translate_for_voice(text, voice) if translate else text

def tokenize_for_voice(text: str, voice: str = 'af_heart', translate: bool = False) -> dict:
    try:
        lang_code = get_voice_lang_code(voice)
        text_to_speak = prepare_text_for_voice(text, voice, translate=translate)
        pipeline = get_quiet_pipeline(lang_code)

        phoneme_chunks = []
        for result in pipeline(text_to_speak):
            if result.phonemes:
                phoneme_chunks.append(result.phonemes)

        phonemes = "\n\n".join(phoneme_chunks).strip()
        return {
            "text": text_to_speak,
            "phonemes": phonemes,
        }
    except Exception as e:
        print(f"Tokenization error for voice {voice}: {e}")
        if voice != 'af_heart':
            return tokenize_for_voice(text, 'af_heart', translate=translate)
        return {
            "text": text,
            "phonemes": "",
        }

def synthesize(text: str, voice: str = 'af_heart', speed: float = 1.0, translate: bool = False) -> bytes:
    try:
        # Determine lang_code
        lang_code = get_voice_lang_code(voice)
        text_to_speak = prepare_text_for_voice(text, voice, translate=translate)
            
        pipeline = get_pipeline(lang_code)
        actual_voice = resolve_voice_path(voice)
        
        generator = pipeline(text_to_speak, voice=actual_voice, speed=speed)
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
            return synthesize(text, 'af_heart', speed, translate=translate)
        return b""

def validate_voice(voice_id: str) -> bool:
    try:
        sample_text = get_voice_settings(voice_id)['health_text']
        audio = synthesize(sample_text, voice_id)
        return len(audio) > 0
    except:
        return False

def get_all_voices() -> dict:
    return VOICE_REGISTRY
