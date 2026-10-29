#!/usr/bin/env python3
"""
Test script for piper_locale server.

Usage:
    # Health check only
    python3 test_piper.py

    # Synthesize test text with a catalog voice (downloads the voice on first use)
    python3 test_piper.py --voice it_IT-paola-medium --lang it

    # Synthesize custom text
    python3 test_piper.py --voice ru_RU-irina-medium --lang ru --text "Привет, странник."
"""

import argparse
import sys

# Windows consoles with a legacy codepage (e.g. CP1251) cannot encode the
# box-drawing characters below; force UTF-8 stdio.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

try:
    import requests
except ImportError:
    print("Missing: pip install requests")
    sys.exit(1)

BASE = "http://127.0.0.1:8007"
DEFAULT_TEXT = "Hello. I am the narrator of this adventure."


def print_section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def health_check():
    print_section("Health")
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        r.raise_for_status()
        d = r.json()
        print(f"  Status    : {d.get('status')}")
        print(f"  Engine    : {d.get('engine')}")
        print(f"  Language  : {d.get('language')}")
        print(f"  Languages : {d.get('languages')}")
        print(f"  Loaded    : {d.get('loaded_voices')}")
        return True
    except requests.exceptions.ConnectionError:
        print("  ERROR: Server not running. Start with: python3 server.py")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def list_voices(lang: str | None):
    print_section(f"Available voices (language={lang or 'server default'})")
    try:
        url = f"{BASE}/voices" + (f"?language={lang}" if lang else "")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        d = r.json()
        print(f"  Language: {d.get('language')}")
        for v in d.get("voices", []):
            print(f"  - {v}")
        return d.get("voices", [])
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def synthesize(text: str, voice: str, lang: str | None):
    print_section("Synthesize")
    out = "test_piper.wav"
    try:
        payload = {"text": text, "voice_id": voice}
        if lang:
            payload["language"] = lang
        r = requests.post(f"{BASE}/synthesize", data=payload, timeout=300)
        r.raise_for_status()
        with open(out, "wb") as f:
            f.write(r.content)
        print(f"  Voice: {voice}")
        print(f"  Text : {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  Saved: {out} ({len(r.content) // 1024} KB)")
        return True
    except requests.exceptions.HTTPError as e:
        print(f"  ERROR {e.response.status_code}: {e.response.text}")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test the piper_locale server")
    parser.add_argument("--voice", help="Catalog voice id (from /voices)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--lang", help="Language code (en ru it fr de es pt zh)")
    args = parser.parse_args()

    if not health_check():
        sys.exit(1)

    voices = list_voices(args.lang)
    if voices and not args.voice:
        print("\nNo --voice given; skipping synthesis (first run downloads the model).")
        sys.exit(0)

    if args.voice:
        if not synthesize(args.text, args.voice, args.lang):
            sys.exit(1)


if __name__ == "__main__":
    main()
