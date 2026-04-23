from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent
LOGO_CANDIDATES = [
    ROOT / "assets" / "logo.png",
    ROOT / "assets" / "logo.jpg",
    ROOT / "logo.png",
    ROOT / "logo.jpg",
]


def _pick_logo() -> Path | None:
    for candidate in LOGO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


LOGO = _pick_logo()
PAGE_ICON = str(LOGO) if LOGO else "🍅"

st.set_page_config(page_title="rAIchica", page_icon=PAGE_ICON, layout="centered", initial_sidebar_state="collapsed")


def _status_label(status: str) -> str:
    return {
        "accepted_diagnosis": "Diagnosis accepted",
        "uncertain_diagnosis": "Uncertain diagnosis",
        "plant_confirmed": "Plant detected",
        "retake_required": "Retake needed",
        "invalid_input": "Invalid input",
        "retake_or_review": "Review needed",
    }.get(status, status.replace("_", " ").title())


def _status_class(status: str) -> str:
    if status == "accepted_diagnosis":
        return "success"
    if status in {"uncertain_diagnosis", "plant_confirmed", "retake_or_review"}:
        return "warning"
    return "danger"


def _result_title(status: str, diagnosis: str) -> str:
    if status == "accepted_diagnosis":
        return f"{diagnosis} detected"
    if status == "uncertain_diagnosis":
        return f"Possible {diagnosis}"
    if status == "plant_confirmed":
        return "Plant detected"
    return _status_label(status)


def _render_result_body(result: dict) -> None:
    status = result.get("status", "unknown")
    diagnosis = result.get("diagnosis", "unknown")
    confidence = float(result.get("confidence", 0.0))
    crop = result.get("crop", "unknown")
    margin = float(result.get("prediction_margin", 0.0))

    st.markdown(f'<div class="status-box {_status_class(status)}">{_status_label(status)}</div>', unsafe_allow_html=True)
    if result.get("message"):
        st.markdown(f'<div class="muted-note">{result["message"]}</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="result-head">
            <div>
                <div class="result-title">{_result_title(status, diagnosis)}</div>
                <div class="result-sub">Crop: <b>{crop}</b></div>
            </div>
            <div class="badge">{confidence:.0%} confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Diagnosis</div>
                <div class="metric-value">{diagnosis}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Margin</div>
                <div class="metric-value">{margin:.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Crop</div>
                <div class="metric-value">{crop}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    advice = result.get("advice_result")
    if advice:
        st.markdown("#### What to do next")
        st.write(advice.get("summary", ""))

        def _render_list(title: str, items: list[str]) -> None:
            if items:
                st.markdown(f"**{title}**")
                st.markdown("\n".join([f"- {item}" for item in items]))

        _render_list("Chemical treatment", advice.get("chemical_treatment", []))
        _render_list("Organic alternatives", advice.get("organic_alternatives", []))
        _render_list("Prevention steps", advice.get("prevention_steps", []))
        _render_list("Neighboring plant protection", advice.get("neighboring_plant_protection", []))

        if advice.get("eco_impact_note"):
            st.markdown("**Eco impact note**")
            st.write(advice["eco_impact_note"])


def _render_result_modal(result: dict) -> None:
    preview = st.session_state.get("latest_preview")
    if preview is not None:
        st.image(preview, caption="Captured leaf", use_container_width=True)
        st.markdown("<div class='hero-note'>Scan complete</div>", unsafe_allow_html=True)
    _render_result_body(result)


if not hasattr(st, "dialog"):
    st.session_state["_dialog_supported"] = False
else:
    st.session_state["_dialog_supported"] = True


if hasattr(st, "dialog"):

    @st.dialog("Scan result", width="large")
    def _result_dialog(result: dict) -> None:
        st.markdown("### Diagnosis result")
        _render_result_modal(result)
        if st.button("Close", type="primary", use_container_width=True):
            st.session_state["show_result_dialog"] = False
            st.rerun()


st.markdown(
    """
    <style>
    :root {
        --bg0: #03110b;
        --bg1: #071913;
        --bg2: #0d2419;
        --panel: rgba(12, 24, 17, 0.88);
        --panel-2: rgba(255,255,255,0.04);
        --border: rgba(255,255,255,0.10);
        --text: #eef9f2;
        --muted: rgba(238,249,242,0.72);
        --muted-2: rgba(238,249,242,0.52);
        --green: #78f0a8;
        --green-strong: #29d27c;
        --amber: #ffd166;
        --red: #ff7d7d;
        --shadow: 0 24px 60px rgba(0,0,0,0.35);
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 8%, rgba(120,240,168,0.20), transparent 24%),
            radial-gradient(circle at 84% 12%, rgba(255,209,102,0.14), transparent 20%),
            linear-gradient(180deg, var(--bg0), var(--bg1) 42%, var(--bg2));
        color: var(--text);
    }

    .block-container {
        max-width: 860px;
        padding-top: 0.9rem;
        padding-bottom: 2rem;
    }

    .hero-shell {
        border: 1px solid var(--border);
        border-radius: 34px;
        background: linear-gradient(180deg, rgba(14,33,22,0.94), rgba(8,18,13,0.92));
        box-shadow: var(--shadow);
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .logo-box img {
        width: 118px;
        height: 118px;
        object-fit: contain;
        border-radius: 28px;
        border: 1px solid rgba(255,255,255,0.10);
        background:
            radial-gradient(circle at 30% 20%, rgba(120,240,168,0.20), transparent 48%),
            rgba(255,255,255,0.03);
        box-shadow: 0 18px 40px rgba(0,0,0,0.25);
    }

    .brand-kicker {
        text-transform: uppercase;
        letter-spacing: 0.22em;
        font-size: 0.72rem;
        color: var(--green);
        font-weight: 900;
        margin-bottom: 0.4rem;
    }

    .brand-title {
        font-size: 2.18rem;
        line-height: 0.98;
        margin: 0;
        font-weight: 950;
        color: var(--text);
    }

    .brand-sub {
        margin-top: 0.55rem;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.5;
        max-width: 58ch;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.9rem;
    }

    .chip {
        padding: 0.42rem 0.7rem;
        border-radius: 999px;
        font-size: 0.8rem;
        color: var(--text);
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
    }

    .panel {
        border: 1px solid var(--border);
        border-radius: 26px;
        background: var(--panel);
        box-shadow: var(--shadow);
        padding: 1rem;
    }

    .panel-title {
        margin: 0 0 0.5rem 0;
        font-size: 1.05rem;
        font-weight: 900;
        color: var(--text);
    }

    .panel-copy {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.5;
    }

    .upload-shell {
        border: 1px dashed rgba(255,255,255,0.14);
        border-radius: 22px;
        background: rgba(255,255,255,0.03);
        padding: 0.9rem;
    }

    .capture-hint {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 0.75rem;
    }

    .capture-pill {
        padding: 0.38rem 0.62rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        color: var(--text);
        font-size: 0.78rem;
        font-weight: 800;
    }

    .hero-note {
        margin-top: 0.9rem;
        color: var(--muted-2);
        font-size: 0.82rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .status-box {
        margin-top: 1rem;
        padding: 0.9rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.10);
        font-weight: 800;
        letter-spacing: 0.01em;
    }

    .status-box.success { background: rgba(39,174,96,0.18); color: #c9f8d8; }
    .status-box.warning { background: rgba(255,209,102,0.15); color: #ffe8a4; }
    .status-box.danger  { background: rgba(255,125,125,0.15); color: #ffd2d2; }

    .result-card {
        margin-top: 1rem;
        border: 1px solid var(--border);
        border-radius: 28px;
        background: rgba(10, 21, 15, 0.88);
        box-shadow: var(--shadow);
        padding: 1rem;
    }

    .result-head {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        align-items: start;
        margin-bottom: 0.8rem;
    }

    .result-title {
        font-size: 1.15rem;
        font-weight: 900;
        color: var(--text);
        margin: 0;
    }

    .result-sub {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.25rem;
        line-height: 1.45;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.38rem 0.66rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 800;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.05);
        color: var(--text);
        white-space: nowrap;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
    }

    .metric {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        background: rgba(255,255,255,0.04);
        padding: 0.85rem;
    }

    .metric-label {
        color: var(--muted-2);
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.24rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.08rem;
        font-weight: 900;
        word-break: break-word;
    }

    .guidance {
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        background: rgba(255,255,255,0.03);
        padding: 0.95rem;
    }

    .guidance-list {
        margin: 0.35rem 0 0 1rem;
        color: var(--text);
    }

    .guidance-list li {
        margin: 0.32rem 0;
    }

    .muted-note {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.45;
    }

    @media (max-width: 720px) {
        .brand-title { font-size: 1.82rem; }
        .metric-grid { grid-template-columns: 1fr; }
        .result-head { flex-direction: column; }
        .logo-box img { width: 92px; height: 92px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

logo = LOGO

st.markdown('<div class="hero-shell">', unsafe_allow_html=True)
intro_left, intro_right = st.columns([1.08, 0.92], gap="medium")

with intro_left:
    with st.container(border=False):
        if logo:
            st.image(str(logo), width=128)
        st.markdown('<div class="brand-kicker">rAIchica</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="brand-title">Your crop, checked in one tap.</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="brand-sub">
                Take a photo or upload an image, and rAIchica tells you what it sees, how confident it is,
                and what to do next. Built to feel fast, calm, and farmer-friendly.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="chip-row">
                <span class="chip">Camera first</span>
                <span class="chip">Upload as backup</span>
                <span class="chip">Local .h5 diagnosis</span>
                <span class="chip">Practical advice</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="hero-note">Built for quick camera capture and fast decisions.</div>', unsafe_allow_html=True)

with intro_right:
    with st.container(border=False):
        st.markdown('<div class="panel-title">Capture</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-copy">Take a fresh photo or pick one from your gallery. The app handles the rest.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="capture-hint">
                <span class="capture-pill">Camera first</span>
                <span class="capture-pill">Upload optional</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        input_mode = st.radio("Input", ["Camera", "Upload"], horizontal=True, label_visibility="collapsed")
        st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
        image_file = st.camera_input("Take a photo") if input_mode == "Camera" else st.file_uploader("Upload plant photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
        analyze = st.button("Analyze plant", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Preview", use_container_width=True)

    if analyze:
        with st.spinner("Reading the leaf..."):
            result = run_pipeline(image)

        st.session_state["latest_result"] = result
        st.session_state["show_result_dialog"] = True
        st.session_state["latest_preview"] = image

        if hasattr(st, "dialog"):
            _result_dialog(result)
            st.session_state["show_result_dialog"] = False
        else:
            with st.container(border=True):
                _render_result_body(result)
else:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Ready for a scan</div>
            <div class="panel-copy">
                Use the camera for the fastest flow, or upload a photo from your gallery.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
