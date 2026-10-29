"""
RpgAi Silero TTS Locale Server
------------------------------
Lightweight CPU-friendly alternative to the XTTS (tts_locale) server: fixed,
model-predefined speakers instead of voice cloning. Quality is lower than
XTTS but synthesis is near-instant on a modern CPU — usable without a GPU.

License note: Silero TTS models are CC BY-NC (non-commercial use only).
Suitable for personal, non-commercial game sessions.

Model: Silero TTS via torch.hub (snakers4/silero-models), loaded lazily per
language on first request and cached in memory. First request for a language
also downloads the model weights (~100-300 MB) into the torch hub cache.

Recommended launch:
  python server.py --host 0.0.0.0

Endpoints (same contract as tts_locale/server.py, so the engine and web UI
work unchanged):
  POST /synthesize
    text     — text to synthesize
    voice_id — speaker name (see GET /voices for the active language)
    language — BCP-47-ish language code (default: server --language)
    Returns: WAV audio bytes (Content-Type: audio/wav)

  GET /voices[?language=xx]  — list speakers for a language
  GET /health                — liveness check
  POST /voices/add           — NOT supported (no cloning); returns 501
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

# python-build-standalone / bare venv interpreters on Windows do not trust the
# system CA store, so downloading model weights fails with a certificate
# verification error. Point urllib at certifi's bundle when it is installed.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import Response

# ============================================================
# ─── CLI ────────────────────────────────────────────────────
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RpgAi Silero TTS Locale Server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host (default: 127.0.0.1, use 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=8006,
                        help="Port (default: 8006)")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu, cuda, ... (default: cpu — the point of this server)")
    parser.add_argument("--language", default="en",
                        help="Default/fallback language (default: en). Supported: ru en de es fr ua uz")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Output sample rate (default: 48000; falls back to 24000/8000 if the model rejects it)")
    return parser.parse_args()

ARGS = parse_args()

# Language → (silero model id, speaker list served by GET /voices).
# 'random' speakers exist in the models but are excluded: an NPC voice must be
# stable across turns, so we only list deterministic speakers.
LANGUAGE_MODELS = {
    "ru": ("v4_ru", ["aidar", "baya", "kseniya", "xenia", "eugene"]),
    "en": ("v3_en", [f"en_{i}" for i in range(118)]),
    "de": ("v3_de", ["bernd_ungerer", "eva_k", "friedrich", "hokuspokus", "karlsson"]),
    "es": ("v3_es", ["es_0", "es_1", "es_2"]),
    "fr": ("v3_fr", [f"fr_{i}" for i in range(6)]),
    "ua": ("v4_ua", ["mykyta"]),
    "uz": ("v3_uz", ["dilnavoz"]),
}

DEVICE = torch.device(ARGS.device)

# Keep the banner plain ASCII: on Windows consoles with a legacy codepage
# (e.g. CP1251) Python stdio cannot encode box-drawing characters and crashes.
print("==============================================")
print("  RpgAi Silero TTS Locale Server")
print(f"  Device:    {DEVICE}")
print(f"  Host:      {ARGS.host}")
print(f"  Port:      {ARGS.port}")
print(f"  Language:  {ARGS.language}")
print(f"  Rate:      {ARGS.sample_rate}")
print("==============================================")

# ============================================================
# ─── MODEL CACHE ────────────────────────────────────────────
# ============================================================
# language → (model, speakers, sample_rate)
_models: dict = {}

def _load_language(language: str):
    """Load and cache the Silero model for a language (torch hub, lazy)."""
    if language in _models:
        return _models[language]
    if language not in LANGUAGE_MODELS:
        supported = ", ".join(sorted(LANGUAGE_MODELS))
        raise HTTPException(400, f"Unsupported language '{language}'. Supported: {supported}")
    model_id, speakers = LANGUAGE_MODELS[language]
    print(f"[startup] Loading silero '{model_id}' on {DEVICE} ...")
    res = torch.hub.load("snakers4/silero-models",
                         model="silero_tts",
                         language=language,
                         speaker=model_id,
                         trust_repo=True)
    # Recent hubconf versions return (model, symbols, sample_rate, ...);
    # older ones returned the model alone.
    hub_rate = None
    if isinstance(res, tuple):
        model = res[0]
        if len(res) > 2 and isinstance(res[2], int):
            hub_rate = res[2]
    else:
        model = res
    model.to(DEVICE)
    sample_rate = _pick_sample_rate(model, hub_rate, speakers[0], language)
    _models[language] = (model, speakers, sample_rate)
    print(f"[startup] '{model_id}' ready (rate={sample_rate}).")
    return _models[language]

def _apply_tts(model, text: str, rate: int, voice: str):
    """apply_tts signature differs across silero generations: current models
    take speaker=, older ones voice=, the oldest also model=."""
    try:
        return model.apply_tts(text=text, sample_rate=rate, speaker=voice)
    except TypeError as e:
        if "speaker" not in str(e):
            raise
    try:
        return model.apply_tts(text=text, sample_rate=rate, voice=voice)
    except TypeError as e:
        if "model" not in str(e):
            raise
    return model.apply_tts(text=text, model=model, sample_rate=rate, voice=voice)

# Word used to probe a working sample rate. Must be valid text in the target
# language: the ru/ua accentuators raise a bare ValueError on Latin-only
# input, which made the probe fail for every rate.
PROBE_TEXT = {"ru": "проверка", "ua": "перевірка"}

def _pick_sample_rate(model, hub_rate, probe_voice, language: str) -> int:
    """Silero models accept 8000/24000/48000; keep the requested rate if it
    works. The probe uses a deterministic speaker — 'random' does not exist
    in every model version (e.g. v4_ru) and would fail the probe."""
    candidates = [r for r in (ARGS.sample_rate, hub_rate, 24000, 8000) if r]
    probe_text = PROBE_TEXT.get(language, "test")
    for rate in candidates:
        try:
            with torch.no_grad():
                probe = _apply_tts(model, probe_text, rate, probe_voice)
            if probe is not None and len(probe) > 0:
                return rate
            print(f"[warn] probe at {rate} Hz returned empty audio")
        except Exception as e:
            print(f"[warn] probe at {rate} Hz failed: {type(e).__name__}: {e}")
    raise RuntimeError("Could not determine a working sample rate for the model")

def _resolve_language(requested: str) -> str:
    """Map a request language to a loaded model language, falling back to the
    server default. Logs the substitution so a misconfigured game language
    (e.g. 'it', which Silero does not cover) is visible instead of silent."""
    lang = (requested or ARGS.language).lower() or ARGS.language
    if lang in LANGUAGE_MODELS:
        return lang
    if lang not in _language_warnings:
        print(f"[warn] language '{lang}' not supported by Silero — falling back to '{ARGS.language}'")
        _language_warnings.add(lang)
    return ARGS.language

_language_warnings: set = set()

app = FastAPI(title="RpgAi Silero TTS Locale Server")

# ============================================================
# ─── POST /synthesize ──────────────────────────────────────
# ============================================================
# Parse the urlencoded body explicitly as UTF-8. FastAPI's Form() goes
# through python-multipart, which falls back to the process locale (e.g.
# cp1251 on a Russian Windows) when the client sends no charset — that
# corrupts any non-ASCII text into mojibake before synthesis.
from urllib.parse import parse_qsl
from fastapi import Request

@app.post("/synthesize")
async def synthesize(request: Request):
    body  = (await request.body()).decode("utf-8", errors="replace")
    form  = dict(parse_qsl(body, keep_blank_values=True))
    text      = form.get("text", "")
    voice_id  = form.get("voice_id", "")
    language  = form.get("language", ARGS.language)
    if not text.strip():
        raise HTTPException(400, "Empty text")
    lang = _resolve_language(language)
    model, speakers, sample_rate = _load_language(lang)
    voice = voice_id if voice_id else speakers[0]
    try:
        with torch.no_grad():
            wav = _apply_tts(model, text, sample_rate, voice)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Synthesis failed: {e}")
    if wav is None or len(wav) == 0:
        raise HTTPException(500, "Synthesis produced empty audio")
    buf = io.BytesIO()
    sf.write(buf, np.array(wav, dtype=np.float32),
             samplerate=sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")

# ============================================================
# ─── GET /voices ───────────────────────────────────────────
# ============================================================
@app.get("/voices")
def list_voices(language: str = ARGS.language):
    lang = _resolve_language(language)
    _, speakers, _ = _load_language(lang)
    return {"voices": speakers, "language": lang}

# ============================================================
# ─── GET /health ───────────────────────────────────────────
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "engine": "silero",
            "device": str(DEVICE), "language": ARGS.language,
            "loaded_languages": sorted(_models.keys()),
            "voices": len(LANGUAGE_MODELS.get(ARGS.language, ("", []))[1])}

# ============================================================
# ─── POST /voices/add ──────────────────────────────────────
# ============================================================
@app.post("/voices/add")
def add_voice():
    raise HTTPException(501, "Silero TTS has fixed model speakers — voice "
                             "cloning/upload is not supported. Use one of the "
                             "speakers from GET /voices.")

# ============================================================
# ─── MAIN ──────────────────────────────────────────────────
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
