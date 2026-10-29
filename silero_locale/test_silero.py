#!/usr/bin/env python3
"""
Test script for silero_locale server.

Usage:
    # Health check only
    python3 test_silero.py

    # Synthesize test text with a speaker of the server language
    python3 test_silero.py --voice aidar

    # Synthesize custom text in another language (model loads lazily)
    python3 test_silero.py --voice en_0 --text "Hello there." --lang en
"""

import argparse
import sys

try:
    import requests
except ImportError:
    print("Missing: pip install requests")
    sys.exit(1)

BASE = "http://127.0.0.1:8006"
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
        print(f"  Status   : {d.get('status')}")
        print(f"  Engine   : {d.get('engine')}")
        print(f"  Device   : {d.get('device')}")
        print(f"  Language : {d.get('language')}")
        print(f"  Loaded   : {d.get('loaded_languages')}")
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
        voices = d.get("voices", [])
        for v in voices:
            print(f"  - {v}")
        return voices
    except Exception as e:
        print(f"  ERROR: {e}")
        return []


def synthesize(text: str, voice: str, lang: str | None):
    print_section("Synthesize")
    out = "test_silero.wav"
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
    parser = argparse.ArgumentParser(description="Test the silero_locale server")
    parser.add_argument("--voice", help="Speaker name (from /voices)")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--lang", help="Language code (ru en de es fr ua uz)")
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
