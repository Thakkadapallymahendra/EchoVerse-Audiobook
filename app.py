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
    load_hf_pipeline,
    rewrite_with_granite,
    tts_with_gtts_to_bytes,
)

st.set_page_config(page_title="EchoVerse", page_icon="🎧", layout="wide")

# ----- Dark minimalist CSS -----
st.markdown("""
<style>
:root{ --bg:#0e1116; --panel:#12171f; --panel2:#171c25; --border:#232a35; --text:#e8eaed; --muted:#9aa0a6; }
html, body, [data-testid="stAppViewContainer"]{ background: var(--bg); }
header{ background: transparent; }
.block-container{ padding-top: 2rem; }
.echotitle{ display:flex; gap:.6rem; align-items:center; justify-content:center; margin-bottom:.25rem; }
.echotitle h1{ margin:0; font-weight:800; letter-spacing:.3px; color:var(--text); }
.caption{ color: var(--muted); font-size:.9rem; }
.echocard{ background: rgba(255,255,255,.03); border:1px solid var(--border); border-radius:14px; padding:14px 16px; }
.stFileUploader, .stSelectbox, .stTextArea, .stTextInput{ background: var(--panel); border:1px solid var(--border); border-radius:12px; padding:8px; }
.stTextArea textarea, .stTextInput input{ color: var(--text); }
.stButton>button{ width:100%; background:#1b212b; color:var(--text); border:1px solid var(--border); border-radius:12px; padding:.6rem 1rem; }
.stButton>button:hover{ background:#202634; border-color:#2a3340; }
section[data-testid="stSidebar"]{ background: var(--panel2); border-right:1px solid var(--border); }
.streamlit-expanderHeader{ color: var(--text); }
</style>
""", unsafe_allow_html=True)

# ----- Sidebar (model + gen settings) -----
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    model_id = st.text_input("HF model", value=os.getenv("HF_MODEL_ID", "ibm-granite/granite-3.2-2b-instruct"))
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    max_new_tokens = st.slider("Max New Tokens", 64, 2048, 512, 32)
    st.caption("Tip: set HF_HOME/TRANSFORMERS_CACHE to control the cache location.")

# ----- Cache the pipeline so it loads once -----
@st.cache_resource(show_spinner=True)
def _get_pipe(mid: str):
    return load_hf_pipeline(mid)

pipe = _get_pipe(model_id)

# ----- Header -----
st.markdown("<div class='echotitle'><span style='font-size:1.4rem'>🎧</span><h1>EchoVerse</h1></div>", unsafe_allow_html=True)

# ----- Voices (gTTS presets) -----
VOICE_PRESETS = {
    "Kate (UK)":   {"lang": "en", "tld": "co.uk", "slow": False},
    "Eric (US)":   {"lang": "en", "tld": "com",   "slow": False},
    "Aditi (EN)":  {"lang": "en", "tld": "co.in", "slow": False},
    "Aditi (HI)":  {"lang": "hi", "tld": "co.in", "slow": False},
    "Sai (TE)":    {"lang": "te", "tld": "co.in", "slow": False},
    "Soft (slow)": {"lang": "en", "tld": "com",   "slow": True},
}

# ----- Input block -----
st.markdown("#### Upload .txt File")
u_col, _ = st.columns([1,1])
with u_col:
    uploaded = st.file_uploader("Drag and drop file here", type=["txt"], label_visibility="collapsed")
st.markdown("<div class='caption'>TXT · up to ~200MB</div>", unsafe_allow_html=True)

st.markdown("#### Or paste your text here:")
text = st.text_area("Input", height=160, label_visibility="collapsed", placeholder="Paste your text here...")

if uploaded is not None:
    try:
        text = uploaded.read().decode("utf-8")
    except Exception:
        text = uploaded.getvalue().decode(errors="ignore")

st.markdown("#### Select Voice")
voice_name = st.selectbox("voice", list(VOICE_PRESETS.keys()), index=0, label_visibility="collapsed")

st.markdown("#### Select Tone")
tone = st.selectbox("tone", DEFAULT_TONES, index=DEFAULT_TONES.index("Suspenseful") if "Suspenseful" in DEFAULT_TONES else 0,
                    label_visibility="collapsed")

c1, c2, c3 = st.columns([1,1,3])
with c1:
    gen = st.button("Generate Audiobook")

# ----- State -----
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

# ----- Generate -----
if gen:
    if not text or not text.strip():
        st.warning("Please provide some input text (upload a .txt or paste text).")
    else:
        try:
            with st.spinner("Rewriting with Granite…"):
                rewritten = rewrite_with_granite(
                    pipe, text.strip(), tone=tone, temperature=temperature, max_new_tokens=max_new_tokens
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
                "model": model_id, "temperature": temperature, "max_new_tokens": max_new_tokens,
                "text_file": str(txt_path), "audio_file": str(mp3_path),
            }
            (outputs / f"meta_{tone_safe}_{ts}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            st.session_state.last_meta = meta

            st.success("Audiobook generated successfully.")
        except Exception as e:
            st.error(str(e))

# ----- Output -----
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
