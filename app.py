from __future__ import annotations

import json

import streamlit as st
from PIL import Image

from src.pipeline import run_pipeline

st.set_page_config(page_title="rAIchica Demo", page_icon="🌿", layout="wide")

st.title("rAIchica - Hybrid AI Crop Diagnosis Demo")
st.caption("Vision validates, local model diagnoses, LLM advises.")

with st.expander("System diagram", expanded=False):
    st.code(
        """User Photo
   ↓
[Streamlit Frontend]
   ↓
[Vision Gatekeeper - GPT-4o Vision]
   ↓
[Decision Layer 1: Input Triage]
   ↓
[Local Disease Model - MobileNetV2]
   ↓
[Decision Layer 2: Diagnostic Trust Policy]
   ↓
[Advice Generator - GPT-4o Mini]
   ↓
[Final Farmer-Friendly Result]""",
        language="text",
    )

st.markdown(
    """
This demo uses a **layered responsibility model**:
1. Vision model checks if the image is usable and extracts crop/symptom context.
2. Local classifier predicts disease and confidence.
3. Trust layer decides accepted vs uncertain diagnosis.
4. Advice model converts outputs into practical agronomy guidance.
"""
)

left, right = st.columns([1.2, 1])

with left:
    uploaded = st.file_uploader("Upload plant photo", type=["jpg", "jpeg", "png", "webp"])
    run = st.button("Run diagnosis", type="primary", use_container_width=True)

with right:
    st.info(
        "For full pipeline with GPT calls, set `OPENAI_API_KEY`. "
        "Without key, app runs in fallback demo mode."
    )

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if run:
        with st.spinner("Running pipeline..."):
            result = run_pipeline(image)

        status = result.get("status", "unknown")
        if status == "accepted_diagnosis":
            st.success(f"Diagnosis accepted: {result.get('diagnosis')} ({result.get('confidence', 0):.2f})")
        elif status in {"uncertain_diagnosis", "retake_or_review"}:
            st.warning("Diagnosis uncertain. Review recommendations below.")
        else:
            st.error(result.get("message", "Input requires retake."))

        if "advice" in result:
            advice = result["advice"]
            st.subheader("Farmer-friendly guidance")
            st.write(advice.get("summary", ""))

            cols = st.columns(2)
            with cols[0]:
                st.markdown("**Chemical treatment**")
                for item in advice.get("chemical_treatment", []):
                    st.write(f"- {item}")

                st.markdown("**Organic alternatives**")
                for item in advice.get("organic_alternatives", []):
                    st.write(f"- {item}")

                st.markdown("**Prevention steps**")
                for item in advice.get("prevention_steps", []):
                    st.write(f"- {item}")

            with cols[1]:
                st.markdown("**Neighboring plant protection**")
                for item in advice.get("neighboring_plant_protection", []):
                    st.write(f"- {item}")

                st.markdown("**Eco impact note**")
                st.write(advice.get("eco_impact_note", ""))

        with st.expander("Debug: full pipeline JSON", expanded=False):
            st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")

else:
    st.caption("Upload a plant image to start.")
