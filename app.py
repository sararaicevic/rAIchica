from __future__ import annotations

import base64
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


def _logo_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


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
        --bg0: #f4f8f1;
        --bg1: #e8f2e2;
        --bg2: #dcebd7;
        --panel: rgba(255,255,255,0.78);
        --panel-2: rgba(255,255,255,0.56);
        --border: rgba(31, 63, 45, 0.10);
        --text: #143325;
        --muted: rgba(20,51,37,0.72);
        --muted-2: rgba(20,51,37,0.52);
        --green: #2f9d63;
        --green-strong: #18784a;
        --amber: #f2b84b;
        --red: #c85d57;
        --shadow: 0 22px 50px rgba(20, 51, 37, 0.12);
    }

    .stApp {
        background:
            radial-gradient(circle at 16% 8%, rgba(47,157,99,0.16), transparent 24%),
            radial-gradient(circle at 84% 12%, rgba(242,184,75,0.18), transparent 20%),
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
        background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(247,251,244,0.92));
        box-shadow: var(--shadow);
        padding: 0.7rem 0.9rem 0.9rem;
        margin-bottom: 1rem;
    }

    .brand-top {
        display: flex;
        justify-content: center;
        margin-bottom: 0.6rem;
    }

    .brand-top img {
        width: 148px;
        height: 148px;
        object-fit: contain;
        border-radius: 32px;
        border: 1px solid rgba(47,157,99,0.12);
        background:
            radial-gradient(circle at 30% 20%, rgba(47,157,99,0.10), transparent 48%),
            rgba(255,255,255,0.9);
        box-shadow: 0 18px 36px rgba(20,51,37,0.11);
    }

    .logo-box img {
        width: 118px;
        height: 118px;
        object-fit: contain;
        border-radius: 28px;
        border: 1px solid rgba(47,157,99,0.12);
        background:
            radial-gradient(circle at 30% 20%, rgba(47,157,99,0.10), transparent 48%),
            rgba(255,255,255,0.85);
        box-shadow: 0 16px 30px rgba(20,51,37,0.10);
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
        color: var(--green-strong);
        background: rgba(47,157,99,0.08);
        border: 1px solid rgba(47,157,99,0.14);
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

    .capture-hint {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 0.75rem;
    }

    .capture-pill {
        padding: 0.38rem 0.62rem;
        border-radius: 999px;
        background: rgba(242,184,75,0.18);
        border: 1px solid rgba(242,184,75,0.24);
        color: #6f4b06;
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
        border: 1px solid rgba(20,51,37,0.08);
        font-weight: 800;
        letter-spacing: 0.01em;
    }

    .status-box.success { background: rgba(47,157,99,0.12); color: #176a42; }
    .status-box.warning { background: rgba(242,184,75,0.16); color: #7b540a; }
    .status-box.danger  { background: rgba(200,93,87,0.12); color: #8e352f; }

    .result-card {
        margin-top: 1rem;
        border: 1px solid var(--border);
        border-radius: 28px;
        background: rgba(255,255,255,0.82);
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
        border: 1px solid rgba(47,157,99,0.16);
        background: rgba(47,157,99,0.08);
        color: var(--green-strong);
        white-space: nowrap;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
    }

    .metric {
        border: 1px solid rgba(20,51,37,0.08);
        border-radius: 20px;
        background: rgba(244,248,241,0.92);
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
        border: 1px solid rgba(20,51,37,0.08);
        border-radius: 24px;
        background: rgba(255,255,255,0.78);
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

    .loading-overlay {
        position: fixed;
        inset: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(244, 248, 241, 0.56);
        backdrop-filter: blur(8px);
    }

    .loading-card {
        width: min(320px, calc(100vw - 2rem));
        padding: 1.15rem 1.1rem;
        border-radius: 28px;
        border: 1px solid rgba(47,157,99,0.12);
        background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(243,248,240,0.96));
        box-shadow: 0 26px 70px rgba(20,51,37,0.14);
        text-align: center;
    }

    .loading-ring {
        width: 74px;
        height: 74px;
        margin: 0 auto 0.9rem;
        border-radius: 50%;
        border: 6px solid rgba(47,157,99,0.10);
        border-top-color: var(--green);
        border-right-color: rgba(242,184,75,0.52);
        animation: loading-spin 0.9s linear infinite;
    }

    .loading-title {
        font-size: 1.02rem;
        font-weight: 900;
        color: var(--text);
        margin: 0;
    }

    .loading-copy {
        margin-top: 0.35rem;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.4;
    }

    div[data-testid="stButton"] button {
        border-radius: 999px !important;
        border: 1px solid rgba(47,157,99,0.22) !important;
        background: linear-gradient(135deg, #d8f1df, #bceacb) !important;
        color: var(--green-strong) !important;
        box-shadow: 0 12px 30px rgba(47,157,99,0.22) !important;
        font-weight: 800 !important;
    }

    div[data-testid="stButton"] button:hover {
        filter: brightness(0.98);
    }

    div[data-testid="stRadio"] label {
        color: var(--text) !important;
    }

    div[data-testid="stRadio"] label * {
        color: var(--text) !important;
    }

    div[data-testid="stRadio"] [data-baseweb="radio"] {
        background: rgba(47,157,99,0.06);
        border: 1px solid rgba(47,157,99,0.14);
        border-radius: 999px;
        padding: 0.14rem 0.4rem;
    }

    div[data-testid="stRadio"] [data-baseweb="radio"] * {
        color: var(--text) !important;
    }

    div[data-testid="stRadio"] [data-baseweb="radio"] svg {
        fill: var(--green) !important;
    }

    div[data-testid="stFileUploader"] * {
        color: var(--text) !important;
        opacity: 1 !important;
    }

    div[data-testid="stCameraInput"] * {
        color: var(--text) !important;
        opacity: 1 !important;
    }

    div[data-testid="stFileUploader"] section,
    div[data-testid="stCameraInput"] section {
        background: rgba(255,255,255,0.92) !important;
        border: 1px solid rgba(47,157,99,0.16) !important;
        border-radius: 18px !important;
    }

    div[data-testid="stFileUploader"] small,
    div[data-testid="stCameraInput"] small,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stCameraInput"] p,
    div[data-testid="stFileUploader"] label,
    div[data-testid="stCameraInput"] label {
        color: var(--text) !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stFileUploader"] button,
    div[data-testid="stCameraInput"] button {
        background: rgba(47,157,99,0.14) !important;
        border: 1px solid rgba(47,157,99,0.22) !important;
        color: var(--text) !important;
    }

    div[data-testid="stDialog"],
    div[role="dialog"] {
        background: rgba(244, 248, 241, 0.58) !important;
    }

    div[data-testid="stDialog"] > div,
    div[role="dialog"] > div {
        background: rgba(255,255,255,0.98) !important;
        color: var(--text) !important;
        border: 1px solid rgba(47,157,99,0.14) !important;
        border-radius: 30px !important;
        box-shadow: 0 28px 80px rgba(20,51,37,0.16) !important;
    }

    div[data-testid="stDialog"] *,
    div[role="dialog"] * {
        color: var(--text) !important;
    }

    div[data-testid="stDialog"] button,
    div[role="dialog"] button {
        background: linear-gradient(135deg, #d8f1df, #bceacb) !important;
        color: var(--green-strong) !important;
    }

    @keyframes loading-spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    @media (max-width: 720px) {
        .brand-title { font-size: 1.82rem; }
        .metric-grid { grid-template-columns: 1fr; }
        .result-head { flex-direction: column; }
        .brand-top img { width: 130px; height: 130px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-shell">', unsafe_allow_html=True)
if LOGO:
    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;margin:0.1rem 0 0.4rem;">
            <img src="{_logo_data_url(LOGO)}" alt="rAIchica logo"
                 style="width:148px;height:148px;object-fit:contain;border-radius:32px;
                        border:1px solid rgba(47,157,99,0.12);
                        background:rgba(255,255,255,0.92);
                        box-shadow:0 18px 36px rgba(20,51,37,0.11);" />
        </div>
        """,
        unsafe_allow_html=True,
    )

intro_left, intro_right = st.columns([1.08, 0.92], gap="medium")

with intro_left:
    with st.container(border=False):
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
                <span class="chip">Fast check</span>
                <span class="chip">Practical advice</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="hero-note">Built for quick camera capture and fast decisions.</div>', unsafe_allow_html=True)

with intro_right:
    with st.container(border=True):
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
        image_file = st.camera_input("Take a photo") if input_mode == "Camera" else st.file_uploader("Upload plant photo", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        analyze = st.button("Analyze plant", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Preview", use_container_width=True)

    if analyze:
        loading_slot = st.empty()
        loading_slot.markdown(
            """
            <div class="loading-overlay">
                <div class="loading-card">
                    <div class="loading-ring"></div>
                    <div class="loading-title">Reading the leaf</div>
                    <div class="loading-copy">Vision checks the image, the model scores it, and advice is prepared.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.spinner("Reading the leaf..."):
            result = run_pipeline(image)
        loading_slot.empty()

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
