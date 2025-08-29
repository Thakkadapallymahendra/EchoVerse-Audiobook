# core_echoverse.py
#!/usr/bin/env python3

import json
import os
import sys
import textwrap
import datetime
from pathlib import Path

import requests

try:
    from gtts import gTTS
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False


DEFAULT_TONES = [
    "Neutral","Suspenseful","Inspiring","Joyful","Calm","Dramatic","Motivational",
    "Humorous","Serious","Urgent","Formal","Casual","Friendly","Authoritative",
    "Romantic","Cinematic","Narrative","Empathetic",
]

def ensure_outputs_dir() -> Path:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_text(text: str, tone: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ensure_outputs_dir()
    safe_tone = "".join(c for c in tone if c.isalnum() or c in ("-", "_")).strip("_")
    path = out_dir / f"rewritten_{safe_tone}_{ts}.txt"
    path.write_text(text, encoding="utf-8")
    return path

# ---------- Ollama helpers ----------
def _ollama_models(base_url: str):
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except requests.RequestException:
        return []

def ensure_model_present(model: str, base_url: str):
    models = _ollama_models(base_url)
    if model not in models:
        msg = (
            f"Ollama model '{model}' not found at {base_url}.\n"
            f"To fix:\n"
            f"  1) Ensure Ollama is running.\n"
            f"  2) Pull the model:\n"
            f"     ollama pull {model}\n"
            f"Installed models: {', '.join(models) if models else '(none detected)'}"
        )
        raise RuntimeError(msg)

def rewrite_with_ollama(
    text: str,
    tone: str,
    model: str = "gemma3:4b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    ensure_model_present(model, base_url)
    url = f"{base_url.rstrip('/')}/api/generate"
    prompt = textwrap.dedent(f"""
    You are a writing assistant.

    Task: Rewrite the user's text in a **{tone}** tone.
    Rules:
    - Preserve the original meaning and key facts.
    - Keep it clear and natural.
    - Maintain the original language (do NOT translate).
    - Use an appropriate register for the tone.
    - Output ONLY the rewritten text—no preface, no quotes, no explanations.

    User text:
    ---
    {text}
    ---
    """).strip()

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
# The 
    try:
        r = requests.post(url, json=payload, timeout=120)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach Ollama at {url}. Error: {e}")

    if r.status_code != 200:
        if r.status_code == 404 and "not found" in r.text.lower():
            raise RuntimeError(
                f"Ollama says the model '{model}' is not found.\n"
                f"Run:  ollama pull {model}\n"
                f"Raw response: {r.text}"
            )
        raise RuntimeError(f"Ollama returned HTTP {r.status_code}: {r.text}")

    data = r.json()
    if "response" not in data:
        raise RuntimeError(f"Unexpected Ollama response: {json.dumps(data)[:500]}")
    return data["response"].strip()

# ---------- gTTS ----------
def tts_with_gtts_to_bytes(text: str, lang: str = "en", tld: str = "com", slow: bool = False) -> bytes:
    if not _HAS_GTTS:
        raise RuntimeError("gTTS not installed. Install with: pip install gTTS")
    import io
    buf = io.BytesIO()
    gTTS(text=text, lang=lang, tld=tld, slow=slow).write_to_fp(buf)
    return buf.getvalue()
