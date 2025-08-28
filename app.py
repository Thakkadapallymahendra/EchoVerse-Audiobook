# app.py
import os
import json
import datetime
from pathlib import Path

import streamlit as st
from core_echoverse import (
    DEFAULT_TONES,
    ensure_outputs_dir,
    save_text,
    rewrite_with_ollama,
    tts_with_gtts_to_bytes,
)

st.set_page_config(page_title="EchoVerse", page_icon="🎧", layout="wide")

st.markdown("""
<style>
:root{
  --bg:#0e1116;                 /* page bg */
  --panel:#12171f;              /* panels */
  --panel-2:#171c25;            /* secondary panels */
  --glass:rgba(255,255,255,.03);
  --border:#232a35;
  --text:#e8eaed;
  --muted:#9aa0a6;
  --focus:#9aa0a6;              /* subtle neutral accent */
}

/* background */
html, body, [data-testid="stAppViewContainer"]{ background: var(--bg); }
header { background: transparent; }

/* container spacing */
.block-container{ padding-top: 2rem; }

/* titles */
.echotitle{ display:flex; gap:.6rem; align-items:center; justify-content:center; margin-bottom:.25rem; }
.echotitle h1{ margin:0; font-weight:800; letter-spacing:.3px; color:var(--text); }
.caption{ color: var(--muted); font-size:.9rem; }

/* cards */
.echocard{
  background: var(--glass);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
}

/* Streamlit widgets: keep neutral, dark */
.stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div, .stFileUploader, .stTextInput input{
  color: var(--text);
}
.stFileUploader, .stSelectbox, .stTextArea, .stTextInput{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px;
}

/* Buttons */
.stButton>button{
  width:100%;
  background: #1b212b;
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: .6rem 1rem;
}
.stButton>button:hover{ background:#202634; border-color:#2a3340; }
.stButton>button:focus{ outline: 2px solid var(--focus); }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: var(--panel-2);
  border-right: 1px solid var(--border);
}

/* Expander */
.streamlit-expanderHeader{ color: var(--text); }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar (settings) ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    ollama_url = st.text_input("Ollama URL", value=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    model = st.text_input("Ollama Model", value=os.getenv("OLLAMA_MODEL", "gemma3:4b"))
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    max_tokens = st.slider("Max Tokens", 64, 2048, 512, 32)
    st.markdown("<div class='caption'>Make sure the model is pulled locally via <code>ollama pull</code>.</div>", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<div class='echotitle'><span style='font-size:1.4rem'>🎧</span><h1>EchoVerse</h1></div>", unsafe_allow_html=True)

# ---------- Voice presets (gTTS only; neutral, no color) ----------
VOICE_PRESETS = {
    "Kate (UK)":   {"lang": "en", "tld": "co.uk", "slow": False},
    "Eric (US)":   {"lang": "en", "tld": "com",   "slow": False},
    "Aditi (EN)":  {"lang": "en", "tld": "co.in", "slow": False},
    "Aditi (HI)":  {"lang": "hi", "tld": "co.in", "slow": False},
    "Sai (TE)":    {"lang": "te", "tld": "co.in", "slow": False},
    "Soft (slow)": {"lang": "en", "tld": "com",   "slow": True},
}

# ---------- Input: upload or paste ----------
st.markdown("#### Upload .txt File")
up_col, _ = st.columns([1,1])
with up_col:
    uploaded = st.file_uploader("Drag and drop file here", type=["txt"], label_visibility="collapsed")
st.markdown("<div class='caption'>TXT · up to ~200MB</div>", unsafe_allow_html=True)

st.markdown("#### Or paste your text here:")
text = st.text_area("Input", height=160, label_visibility="collapsed", placeholder="Paste your text here...")

if uploaded is not None:
    try:
        text = uploaded.read().decode("utf-8")
    except Exception:
        text = uploaded.getvalue().decode(errors="ignore")

# ---------- Selections ----------
st.markdown("#### Select Voice")
voice_name = st.selectbox("voice", list(VOICE_PRESETS.keys()), index=0, label_visibility="collapsed")

st.markdown("#### Select Tone")
tone = st.selectbox("tone", DEFAULT_TONES, index=DEFAULT_TONES.index("Suspenseful") if "Suspenseful" in DEFAULT_TONES else 0,
                    label_visibility="collapsed")

# ---------- Action button ----------
c1, c2, c3 = st.columns([1,1,3])
with c1:
    gen = st.button("Generate Audiobook")

if "rewritten" not in st.session_state:
    st.session_state.rewritten = ""
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = b""
if "audio_mime" not in st.session_state:
    st.session_state.audio_mime = "audio/mp3"
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}

def _safe_name(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-","_")).strip("_")

if gen:
    if not text or not text.strip():
        st.warning("Please provide some input text (upload a .txt or paste text).")
    else:
        try:
            with st.spinner("Rewriting with Ollama…"):
                rewritten = rewrite_with_ollama(
                    text.strip(),
                    tone=tone,
                    model=model,
                    base_url=ollama_url,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            st.session_state.rewritten = rewritten

            v = VOICE_PRESETS[voice_name]
            with st.spinner("Generating audio with gTTS…"):
                audio_bytes = tts_with_gtts_to_bytes(rewritten, lang=v["lang"], tld=v["tld"], slow=v["slow"])
            st.session_state.audio_bytes = audio_bytes
            st.session_state.audio_mime = "audio/mp3"

            outputs = ensure_outputs_dir()
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tone_safe = _safe_name(tone)
            txt_path = save_text(rewritten, tone)
            mp3_path = outputs / f"speech_{tone_safe}_{ts}.mp3"
            mp3_path.write_bytes(audio_bytes)

            meta = {
                "timestamp": ts, "tone": tone, "voice": voice_name,
                "model": model, "ollama_url": ollama_url,
                "temperature": temperature, "max_tokens": max_tokens,
                "text_file": str(txt_path), "audio_file": str(mp3_path)
            }
            (outputs / f"meta_{tone_safe}_{ts}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            st.session_state.last_meta = meta

            st.success("Audiobook generated successfully.")
        except Exception as e:
            st.error(str(e))

# ---------- Output ----------
if st.session_state.rewritten:
    st.markdown("### Original vs Rewritten Text")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original Text**")
        st.markdown(f"<div class='echocard'>{text}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Rewritten Text**")
        st.markdown(f"<div class='echocard'>{st.session_state.rewritten}</div>", unsafe_allow_html=True)

if st.session_state.audio_bytes:
    st.markdown("### Listen to Your Audiobook")
    st.audio(st.session_state.audio_bytes, format=st.session_state.audio_mime)

    outputs = ensure_outputs_dir()
    ts = st.session_state.last_meta.get("timestamp", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tone_safe = _safe_name(tone)
    st.download_button(
        "Download MP3",
        data=st.session_state.audio_bytes,
        file_name=f"speech_{tone_safe}_{ts}.mp3",
        mime="audio/mp3"
    )

with st.expander("View Past Narrations"):
    out = ensure_outputs_dir()
    files = sorted(out.glob("speech_*.mp3"), reverse=True)
    if not files:
        st.caption("No previous narrations yet.")
    else:
        for f in files[:20]:
            meta_path = out / f"meta_{'_'.join(f.stem.split('_')[1:])}.json"
            col_a, col_b = st.columns([3,1])
            with col_a:
                st.write(f"**{f.name}**")
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        st.caption(f"Tone: {meta.get('tone')} · Voice: {meta.get('voice')} · Model: {meta.get('model')} · {meta.get('timestamp')}")
                    except Exception:
                        pass
            with col_b:
                st.download_button("Download", data=f.read_bytes(), file_name=f.name, mime="audio/mp3", use_container_width=True)
