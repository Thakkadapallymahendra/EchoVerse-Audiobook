# core_echoverse.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import re
import json
import datetime
from pathlib import Path
import textwrap

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

try:
    from gtts import gTTS
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False


MODEL_ID = os.getenv("HF_MODEL_ID", "ibm-granite/granite-3.2-2b-instruct")

DEFAULT_TONES = [
    "Neutral","Suspenseful","Inspiring","Joyful","Calm","Dramatic","Motivational",
    "Humorous","Serious","Urgent","Formal","Casual","Friendly","Authoritative",
    "Romantic","Cinematic","Narrative","Empathetic",
]


# ---------- Files ----------
def ensure_outputs_dir() -> Path:
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    return out

def save_text(text: str, tone: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ensure_outputs_dir()
    safe_tone = "".join(c for c in tone if c.isalnum() or c in ("-","_")).strip("_")
    p = out / f"rewritten_{safe_tone}_{ts}.txt"
    p.write_text(text, encoding="utf-8")
    return p


# ---------- LLM: Granite via Hugging Face ----------
def _pick_device_map_and_dtype():
    """
    Decide device + dtype:
      - CUDA: float16 (or 4-bit if bitsandbytes present and env says so)
      - MPS (Apple Silicon): float16
      - CPU: float32
    """
    use_4bit = os.getenv("HF_USE_4BIT", "false").lower() in ("1","true","yes")
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
        quant = "4bit" if use_4bit else None
        return device_map, dtype, quant
    if torch.backends.mps.is_available():
        return {"": "mps"}, torch.float16, None
    return "cpu", torch.float32, None

def load_hf_pipeline(model_id: str = MODEL_ID):
    device_map, dtype, quant = _pick_device_map_and_dtype()
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if quant == "4bit":
        try:
            from bitsandbytes import __version__ as _  # noqa: F401
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_4bit=True,
                device_map=device_map,
                torch_dtype=torch.float16,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=dtype,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
        )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        device_map=device_map,
        torch_dtype=dtype,
    )
    return gen

def _compose_prompt(text: str, tone: str) -> str:
    # Strong, explicit instruction to avoid any labels/fences
    return textwrap.dedent(f"""
    You are a writing assistant.

    Task: Rewrite the user's text in a **{tone}** tone.

    Output rules (very important):
    - Output ONLY the rewritten text content.
    - DO NOT include any labels like "Rewritten:", "Rewritten text:", "Output:", etc.
    - DO NOT include code fences, markers, or separators such as ``` or ---.
    - DO NOT quote the text with leading/trailing quotation marks.
    - Preserve the original meaning and key facts.
    - Keep it clear and natural.
    - Maintain the original language (do NOT translate).
    - Use an appropriate register for the tone.

    User text:
    {text}
    """).strip()

def _postprocess(s: str) -> str:
    """
    Remove any stray labels, fences, and wrapping quotes the model may add.
    """
    s = s.strip()

    # Remove triple-backtick code fences (start or end)
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s, flags=re.DOTALL)  # ``` or ```lang
    s = re.sub(r"\s*```$", "", s)

    # Remove leading "Rewritten text:", "Rewritten:", "Output:", "Answer:", "Response:", etc (case-insensitive)
    s = re.sub(
        r"^(?:rewritten(?:\s+text)?|output|answer|response|final|result)\s*:?\s*",
        "",
        s,
        flags=re.IGNORECASE,
    )

    # Remove leading or trailing fence-like lines of --- — – or *** etc.
    s = re.sub(r"^(?:[-–—*_`]{3,}\s*)+", "", s)
    s = re.sub(r"(?:\s*[-–—*_`]{3,})+$", "", s)

    # Clean common leading punctuation/noise
    s = re.sub(r"^[>\-\–\—:\s]+", "", s)

    # Strip matching surrounding quotes if the whole thing is quoted
    if len(s) >= 2 and s[0] in "\"'“”‘’" and s[-1] in "\"'“”‘’":
        s = s[1:-1].strip()

    # Remove any leftover double newlines at the start/end
    s = s.strip()

    return s

def rewrite_with_granite(pipe, text: str, tone: str,
                         temperature: float = 0.7, max_new_tokens: int = 512) -> str:
    prompt = _compose_prompt(text, tone)
    out = pipe(
        prompt,
        do_sample=True,
        temperature=float(temperature),
        top_p=0.9,
        repetition_penalty=1.05,
        max_new_tokens=int(max_new_tokens),
        return_full_text=False,  # just the continuation
        eos_token_id=pipe.tokenizer.eos_token_id,
    )
    text_out = out[0]["generated_text"]
    return _postprocess(text_out)


# ---------- gTTS ----------
def tts_with_gtts_to_bytes(text: str, lang: str = "en", tld: str = "com", slow: bool = False) -> bytes:
    if not _HAS_GTTS:
        raise RuntimeError("gTTS not installed. Install with: pip install gTTS")
    buf = io.BytesIO()
    gTTS(text=text, lang=lang, tld=tld, slow=slow).write_to_fp(buf)
    return buf.getvalue()
