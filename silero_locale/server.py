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

Text handling:
- 'ru' requests: Latin characters are transliterated to Cyrillic (John →
  Джон) — the models' alphabet filter would otherwise drop them.
- Unsupported languages (e.g. 'it' — Silero has no Italian model) fall back
  to the server's --language with accented Latin folded to ASCII ("Così" →
  "Cosi"), so an approximation is still voiced; Italian text additionally
  gets a phonetic pass for the English fallback ("Ciao" → "Chiao"). The
  substitution is logged.
- If the model's alphabet filter leaves nothing to synthesize, /synthesize
  returns 422 with an explanatory message (use the 'xtts' provider for
  languages outside Silero's set: ru en de es fr ua uz).

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
import unicodedata

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

# ---------------------------------------------------------------------------
# Latin → Cyrillic transliteration for 'ru' requests. Silero's alphabet filter
# silently DROPS Latin letters from Russian text (and rejects Latin-only text
# outright), so names like "John" would vanish from the audio. Transliterating
# them keeps the words voiced ("Джон").
# ---------------------------------------------------------------------------
_RU_MULTI = [("shch", "щ"), ("sch", "щ"), ("zh", "ж"), ("kh", "х"), ("yo", "ё"),
             ("ts", "ц"), ("ch", "ч"), ("sh", "ш"), ("yu", "ю"), ("ya", "я"),
             ("ju", "ю"), ("ja", "я")]
_RU_SINGLE = dict(a="а", b="б", c="к", d="д", e="е", f="ф", g="г", h="х", i="и",
                  j="дж", k="к", l="л", m="м", n="н", o="о", p="п", q="к", r="р",
                  s="с", t="т", u="у", v="в", w="в", x="кс", y="й", z="з")

def transliterate_ru(text: str) -> str:
    out = []
    i = 0
    lower = text.lower()
    while i < len(text):
        for pair, cyr in _RU_MULTI:
            if lower.startswith(pair, i):
                out.append(cyr.upper() if text[i:i+len(pair)].isupper() else cyr)
                i += len(pair)
                break
        else:
            ch = lower[i]
            if ch in _RU_SINGLE:
                out.append(_RU_SINGLE[ch].upper() if text[i].isupper() else _RU_SINGLE[ch])
            else:
                out.append(text[i])
            i += 1
    return "".join(out)

# Accented Latin → plain ASCII, used when an unsupported language (e.g. 'it',
# which has no Silero model) falls back to a Latin-alphabet model: "Così
# parlò" becomes "Cosi parlo" so at least *something* gets voiced. The
# pronunciation follows the fallback language, not the original — logged.
_ACCENTS = {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "å": "a", "Æ": "Ae",
            "Œ": "Oe", "Ø": "O", "Å": "A", "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th"}

def fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    folded = "".join(c for c in decomposed if not unicodedata.combining(c))
    for k, v in _ACCENTS.items():
        folded = folded.replace(k, v)
    return folded

# Italian → English phonetic transliteration, applied when an 'it' request
# falls back to a Latin-alphabet model. Accent folding alone keeps every
# letter voiced but the English model would read it with English phonetics
# ("ciao" as "kiao"); these digraph rules make the approximation recognisably
# Italian ("ciao" → "chiao", "lasagne" → "lasanye", "perché" → "perke").
# Single-pass longest-match so a rule's output is never rescanned ("ghi"
# must stay "gi", not become "ji").
_IT_PAIRS = [
    ("gli", "ly"),
    ("gn", "ny"),
    ("ghe", "ge"), ("ghi", "gi"),
    ("che", "ke"), ("chi", "ki"),
    ("sce", "she"), ("sci", "shi"),
    ("cia", "cha"), ("cie", "che"), ("cio", "cho"), ("ciu", "chu"),
    ("gia", "ja"), ("gie", "je"), ("gio", "jo"), ("giu", "ju"),
    ("ge", "je"), ("gi", "ji"),
    ("ce", "che"), ("ci", "chi"),
]

def _emit_like(matched: str, out: str) -> str:
    if matched.isupper():
        return out.upper()
    if matched[:1].isupper():
        return out.capitalize()
    return out

def transliterate_it_en(text: str) -> str:
    out = []
    i = 0
    lower = text.lower()
    while i < len(text):
        for pair, en in _IT_PAIRS:
            if lower.startswith(pair, i):
                out.append(_emit_like(text[i:i+len(pair)], en))
                i += len(pair)
                break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)

def _supported_list() -> str:
    return ", ".join(sorted(LANGUAGE_MODELS))

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
    server default. Logs the substitution (once per language) so a
    misconfigured game language — e.g. 'it', which Silero does not cover —
    is visible instead of silent."""
    lang = (requested or ARGS.language).lower() or ARGS.language
    if lang in LANGUAGE_MODELS:
        return lang
    if lang not in _language_warnings:
        print(f"[warn] language '{lang}' is not supported by Silero "
              f"(supported: {_supported_list()}) — falling back to '{ARGS.language}'; "
              f"accents will be folded but pronunciation will follow the fallback language. "
              f"For full quality use the 'xtts' provider, which covers 17 languages.")
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
    requested = (language or ARGS.language).lower()
    lang = _resolve_language(language)
    fell_back = lang != requested
    if fell_back:
        # Unsupported original language: fold accented Latin to plain ASCII so
        # the fallback model voices at least an approximation ("Cosi parlo").
        # Italian gets an extra phonetic pass so the approximation is
        # recognisably Italian ("Ciao" -> "Chiao").
        text = fold_accents(text)
        if requested == "it":
            text = transliterate_it_en(text)
    elif lang == "ru":
        # Russian game: Latin names/words would be dropped by the alphabet
        # filter — transliterate them to Cyrillic instead.
        text = transliterate_ru(text)
    model, speakers, sample_rate = _load_language(lang)
    voice = voice_id if voice_id else speakers[0]
    try:
        with torch.no_grad():
            wav = _apply_tts(model, text, sample_rate, voice)
    except ValueError as e:
        if str(e) == "":
            # The model's alphabet filter removed every character.
            raise HTTPException(422,
                f"Text has no characters the '{lang}' model can synthesize. "
                f"Silero supports: {_supported_list()}. For other languages "
                f"(e.g. Italian) use the 'xtts' TTS provider instead of 'silero'.")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Synthesis failed: {e}")
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
