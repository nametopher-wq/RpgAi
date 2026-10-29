"""
RpgAi Piper TTS Locale Server
-----------------------------
Lightweight CPU TTS provider based on Piper (VITS + espeak-ng phonemization,
ONNX runtime). Unlike the silero models, espeak phonemizes each language
natively, so no alphabet-filter surprises and no transliteration hacks —
Italian/Portuguese/Chinese text just works with a voice of the same locale.

License note: Piper voices are MIT licensed (rhasspy/piper-voices).

Voices are downloaded lazily from the rhasspy/piper-voices HuggingFace repo
on first use and cached under piper_locale/models/. GET /voices lists the
catalog without downloading anything.

Recommended launch:
  python server.py --host 0.0.0.0

Endpoints (same contract as tts_locale/silero_locale, engine + web UI
work unchanged):
  POST /synthesize
    text     — text to synthesize
    voice_id — catalog id, e.g. "it_IT-paola-medium" (see GET /voices)
    language — language of the text (default: server --language)
    Returns: WAV audio bytes (Content-Type: audio/wav)

  GET /voices[?language=xx]  — catalog voices for a language (no download)
  GET /health                — liveness check
  POST /voices/add           — NOT supported (no cloning); returns 501

Strict language handling: an unsupported language (e.g. 'ja', which Piper
does not cover) returns 422 naming the supported set and pointing at the
'xtts' provider — no silent fallback to a wrong-locale voice.
"""

import argparse
import io
import os
import sys

# Windows consoles with a legacy codepage (e.g. CP1251) cannot encode Cyrillic
# log output; force UTF-8 stdio so prints never crash the request handler.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from urllib.parse import parse_qsl

# ============================================================
# ─── CLI ────────────────────────────────────────────────────
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RpgAi Piper TTS Locale Server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host (default: 127.0.0.1, use 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=8007,
                        help="Port (default: 8007)")
    parser.add_argument("--language", default="en",
                        help="Default language (default: en)")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
                        help="Voice model cache directory (default: piper_locale/models/)")
    return parser.parse_args()

ARGS = parse_args()
os.makedirs(ARGS.data_dir, exist_ok=True)

# Voice catalog from rhasspy/piper-voices (v1.0.0). voice_id doubles as the
# file name: <locale>-<voice>-<quality>. Only a curated subset — enough for a
# distinct narrator + a few NPC voices per game language.
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"

CATALOG = {
    "en": [("en_US-lessac-medium", "en_US", "lessac", "medium"),
           ("en_US-ryan-high",     "en_US", "ryan",   "high"),
           ("en_US-amy-medium",    "en_US", "amy",    "medium"),
           ("en_GB-alan-medium",   "en_GB", "alan",   "medium")],
    "ru": [("ru_RU-irina-medium",  "ru_RU", "irina",  "medium"),
           ("ru_RU-dmitri-medium", "ru_RU", "dmitri", "medium"),
           ("ru_RU-ruslan-medium", "ru_RU", "ruslan", "medium"),
           ("ru_RU-denis-medium",  "ru_RU", "denis",  "medium")],
    "it": [("it_IT-paola-medium",  "it_IT", "paola",  "medium"),
           ("it_IT-riccardo-x_low", "it_IT", "riccardo", "x_low")],
    "fr": [("fr_FR-siwis-medium",  "fr_FR", "siwis",  "medium"),
           ("fr_FR-gilles-low",    "fr_FR", "gilles", "low"),
           ("fr_FR-upmc-medium",   "fr_FR", "upmc",   "medium")],
    "de": [("de_DE-thorsten-medium", "de_DE", "thorsten", "medium"),
           ("de_DE-eva_k-x_low",   "de_DE", "eva_k",  "x_low"),
           ("de_DE-karlsson-low",  "de_DE", "karlsson", "low")],
    "es": [("es_ES-davefx-medium", "es_ES", "davefx", "medium"),
           ("es_MX-ald-medium",    "es_MX", "ald",    "medium")],
    "pt": [("pt_BR-faber-medium",  "pt_BR", "faber",  "medium"),
           ("pt_BR-cadu-medium",   "pt_BR", "cadu",   "medium"),
           ("pt_PT-tugão-medium",  "pt_PT", "tugão",  "medium")],
    "zh": [("zh_CN-huayan-medium", "zh_CN", "huayan", "medium"),
           ("zh_CN-huayan-x_low",  "zh_CN", "huayan", "x_low")],
}

print("==============================================")
print("  RpgAi Piper TTS Locale Server")
print(f"  Host:      {ARGS.host}")
print(f"  Port:      {ARGS.port}")
print(f"  Language:  {ARGS.language}")
print(f"  Data dir:  {ARGS.data_dir}")
print("==============================================")

# ============================================================
# ─── VOICE LOADING ──────────────────────────────────────────
# ============================================================
from piper import PiperVoice
import requests

_voices: dict = {}  # voice_id → PiperVoice

def _download_voice(voice_id: str, locale: str, voice: str, quality: str):
    """Fetch the .onnx model + config into the cache dir (idempotent).
    Retries each file: the HF CDN intermittently fails the TLS handshake
    from this network, and a fresh attempt usually lands on a healthy edge."""
    lang_dir = locale.split("_")[0]
    base = f"{HF_BASE}/{lang_dir}/{locale}/{voice}/{quality}"
    for suffix in (".onnx", ".onnx.json"):
        dest = os.path.join(ARGS.data_dir, voice_id + suffix)
        if os.path.exists(dest):
            continue
        url = f"{base}/{voice_id}{suffix}"
        last_err = None
        for attempt in range(4):
            print(f"[download] {url} (attempt {attempt + 1})")
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(dest + ".part", "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                    os.replace(dest + ".part", dest)  # atomic: never a half file
                last_err = None
                break
            except Exception as e:
                last_err = e
                try: os.unlink(dest + ".part")
                except OSError: pass
                import time
                time.sleep(1.5 * (attempt + 1))
        if last_err is not None:
            raise HTTPException(502, f"Failed to download voice '{voice_id}': {last_err}")

def _load_voice(voice_id: str):
    if voice_id in _voices:
        return _voices[voice_id]
    entry = next((e for lang in CATALOG.values() for e in lang if e[0] == voice_id), None)
    if not entry:
        known = ", ".join(sorted(v for lang in CATALOG.values() for v, *_ in lang))
        raise HTTPException(404, f"Unknown voice '{voice_id}'. Available: {known}")
    _, locale, voice, quality = entry
    _download_voice(voice_id, locale, voice, quality)
    print(f"[startup] Loading piper voice '{voice_id}' ...")
    v = PiperVoice.load(os.path.join(ARGS.data_dir, voice_id + ".onnx"))
    _voices[voice_id] = v
    print(f"[startup] '{voice_id}' ready.")
    return v

def _synth_audio(voice, text: str):
    """Collect synthesis output: piper-tts' synthesize() yields one
    AudioChunk (audio_float_array + sample_rate) per sentence."""
    arrays = []
    rate = None
    for chunk in voice.synthesize(text):
        arr = getattr(chunk, "audio_float_array", None)
        if arr is not None:
            arrays.append(np.asarray(arr, dtype=np.float32))
        sr = getattr(chunk, "sample_rate", None)
        if sr:
            rate = sr
    if not arrays:
        raise RuntimeError("Piper produced no audio")
    return np.concatenate(arrays), (rate or getattr(voice.config, "sample_rate", 22050))

def _resolve(requested: str):
    """Strict language lookup — no silent wrong-locale fallback."""
    lang = (requested or ARGS.language).lower() or ARGS.language
    if lang not in CATALOG:
        raise HTTPException(422,
            f"Language '{lang}' is not supported by Piper. Supported: "
            f"{', '.join(sorted(CATALOG))}. For other languages (e.g. 'ja') "
            f"use the 'xtts' TTS provider instead of 'piper'.")
    return lang

app = FastAPI(title="RpgAi Piper TTS Locale Server")

# ============================================================
# ─── POST /synthesize ──────────────────────────────────────
# ============================================================
@app.post("/synthesize")
async def synthesize(request: Request):
    # Parse the urlencoded body explicitly as UTF-8 — same reason as in
    # silero_locale: python-multipart falls back to the process locale.
    body  = (await request.body()).decode("utf-8", errors="replace")
    form  = dict(parse_qsl(body, keep_blank_values=True))
    text      = form.get("text", "")
    voice_id  = form.get("voice_id", "")
    language  = form.get("language", ARGS.language)
    if not text.strip():
        raise HTTPException(400, "Empty text")
    lang = _resolve(language)
    if not voice_id:
        voice_id = CATALOG[lang][0][0]
    elif not any(voice_id == e[0] for e in CATALOG[lang]):
        raise HTTPException(400, f"Voice '{voice_id}' is not a '{lang}' voice. "
                                 f"Use one of: {', '.join(v for v, *_ in CATALOG[lang])}")
    voice = _load_voice(voice_id)
    try:
        wav, rate = _synth_audio(voice, text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Synthesis failed: {e}")
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32),
             samplerate=rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")

# ============================================================
# ─── GET /voices ───────────────────────────────────────────
# ============================================================
@app.get("/voices")
def list_voices(language: str = ARGS.language):
    lang = _resolve(language)
    return {"voices": [v for v, *_ in CATALOG[lang]], "language": lang}

# ============================================================
# ─── GET /health ───────────────────────────────────────────
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "engine": "piper", "language": ARGS.language,
            "languages": sorted(CATALOG.keys()),
            "loaded_voices": sorted(_voices.keys())}

# ============================================================
# ─── POST /voices/add ──────────────────────────────────────
# ============================================================
@app.post("/voices/add")
def add_voice():
    raise HTTPException(501, "Piper TTS uses fixed pretrained voices — "
                             "voice cloning/upload is not supported. Pick one "
                             "from GET /voices.")

# ============================================================
# ─── MAIN ──────────────────────────────────────────────────
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
