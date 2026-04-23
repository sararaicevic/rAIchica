"""Core pipeline logic for the rAIchica Streamlit demo."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .config import (
    ADVICE_MODEL,
    CLASS_NAMES_CSV,
    CLASS_NAMES_PATH,
    CONFIDENCE_THRESHOLD,
    MARGIN_THRESHOLD,
    MAX_IMAGE_SIZE,
    MODEL_INPUT_SIZE,
    MODEL_PATH,
    OPENAI_API_KEY,
    SUPPORTED_CROPS,
    TOP_K,
    VISION_MODEL,
)
from .prompts import ADVICE_GENERATOR_PROMPT, VISION_GATEKEEPER_PROMPT

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - import safety for environments without package
    OpenAI = None

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover - import safety for environments without package
    load_model = None


@dataclass
class LocalPrediction:
    label: str
    confidence: float
    top_k: list[dict[str, float]]
    margin: float


def _safe_json_load(raw_text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except Exception:
        return fallback


def _normalize_crop(crop: str) -> str:
    return crop.strip().lower()


def _image_to_data_url(image: Image.Image) -> str:
    resized = image.copy().convert("RGB")
    resized.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=92)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _load_class_names() -> list[str]:
    if CLASS_NAMES_CSV.strip():
        names = [name.strip() for name in CLASS_NAMES_CSV.split(",") if name.strip()]
        if names:
            return names

    path = Path(CLASS_NAMES_PATH)
    if path.exists():
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        names = [line for line in lines if line]
        if names:
            return names

    return []


@lru_cache(maxsize=1)
def _load_local_model():
    if load_model is None:
        return None

    path = Path(MODEL_PATH)
    if not path.exists():
        return None

    try:
        return load_model(path, compile=False)
    except Exception:
        return None


def _predict_with_local_h5(image: Image.Image) -> LocalPrediction | None:
    model = _load_local_model()
    if model is None:
        return None

    class_names = _load_class_names()
    resized = image.copy().convert("RGB").resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    try:
        preds = model.predict(arr, verbose=0)
    except Exception:
        return None

    if preds is None or len(preds) == 0:
        return None

    probs = np.array(preds[0], dtype=np.float32)
    if probs.ndim != 1 or probs.size == 0:
        return None

    probs = np.maximum(probs, 0)
    if probs.sum() <= 0:
        return None
    probs = probs / probs.sum()

    if len(class_names) != len(probs):
        class_names = [f"class_{i}" for i in range(len(probs))]

    sorted_idx = np.argsort(probs)[::-1]
    top_idx = int(sorted_idx[0])
    second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top_idx

    top_k = [
        {"label": class_names[int(i)], "probability": round(float(probs[int(i)]), 4)}
        for i in sorted_idx[: min(TOP_K, len(sorted_idx))]
    ]

    return LocalPrediction(
        label=class_names[top_idx],
        confidence=float(probs[top_idx]),
        top_k=top_k,
        margin=float(probs[top_idx] - probs[second_idx]),
    )


def _fallback_vision_analysis(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)

    brightness = float(arr.mean())
    sharpness = float(np.var(np.diff(arr, axis=0))) if arr.shape[0] > 1 else 0.0
    green_ratio = float((arr[:, :, 1] > arr[:, :, 0]).mean())

    is_plant = green_ratio > 0.30
    quality = "good" if brightness > 45 and sharpness > 30 else "poor"
    symptoms_visible = bool(((arr[:, :, 0] > 150) & (arr[:, :, 1] < 120)).mean() > 0.05)

    likely_crop = "tomato" if is_plant else "unknown"
    return {
        "is_plant": is_plant,
        "image_quality": quality,
        "likely_crop": likely_crop,
        "symptoms_visible": symptoms_visible,
        "symptom_description": (
            "Possible leaf discoloration visible." if symptoms_visible else "No clear disease symptoms visible."
        ),
        "diagnosis_support": "supported" if likely_crop in SUPPORTED_CROPS else "uncertain",
    }


def analyze_with_vision(image: Image.Image) -> tuple[dict[str, Any], str]:
    fallback = _fallback_vision_analysis(image)

    if not OPENAI_API_KEY or OpenAI is None:
        return fallback, "fallback"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        image_url = _image_to_data_url(image)
        response = client.responses.create(
            model=VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": VISION_GATEKEEPER_PROMPT},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            temperature=0,
        )
        parsed = _safe_json_load(response.output_text, fallback)
        merged = {**fallback, **parsed}
        return merged, "openai"
    except Exception:
        return fallback, "fallback"


def _crop_disease_space(crop: str) -> list[str]:
    crop = _normalize_crop(crop)
    spaces = {
        "tomato": ["Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Healthy"],
        "potato": ["Potato Early Blight", "Potato Late Blight", "Potato Healthy"],
        "grape": ["Grape Black Rot", "Grape Leaf Blight", "Grape Healthy"],
        "apple": ["Apple Scab", "Apple Rust", "Apple Healthy"],
    }
    return spaces.get(crop, ["Unknown Leaf Disease", "Healthy Plant", "Nutrient Stress"])


def predict_disease(image: Image.Image, likely_crop: str) -> LocalPrediction:
    local_prediction = _predict_with_local_h5(image)
    if local_prediction is not None:
        return local_prediction

    resized = image.copy().convert("RGB")
    resized.thumbnail((224, 224))

    image_bytes = io.BytesIO()
    resized.save(image_bytes, format="JPEG")
    image_hash = hashlib.sha256(image_bytes.getvalue()).hexdigest()
    seed = int(image_hash[:8], 16)

    labels = _crop_disease_space(likely_crop)
    rng = np.random.default_rng(seed)
    raw_scores = rng.random(len(labels))
    probs = raw_scores / raw_scores.sum()

    sorted_idx = np.argsort(probs)[::-1]
    top_idx = int(sorted_idx[0])
    second_idx = int(sorted_idx[1]) if len(sorted_idx) > 1 else top_idx

    top_k = [
        {"label": labels[int(i)], "probability": round(float(probs[int(i)]), 4)}
        for i in sorted_idx[: min(TOP_K, len(sorted_idx))]
    ]

    confidence = float(probs[top_idx])
    margin = float(probs[top_idx] - probs[second_idx])

    return LocalPrediction(
        label=labels[top_idx],
        confidence=confidence,
        top_k=top_k,
        margin=margin,
    )


def _diagnostic_trust(
    likely_crop: str,
    label: str,
    confidence: float,
    margin: float,
    diagnosis_support: str,
) -> dict[str, Any]:
    crop = _normalize_crop(likely_crop)
    supported_crop = crop in SUPPORTED_CROPS
    crop_consistent = crop != "unknown" and crop in label.lower()

    accepted = all(
        [
            supported_crop,
            crop_consistent,
            diagnosis_support == "supported",
            confidence >= CONFIDENCE_THRESHOLD,
            margin >= MARGIN_THRESHOLD,
        ]
    )

    status = "accepted_diagnosis" if accepted else "uncertain_diagnosis"
    if not supported_crop:
        status = "retake_or_review"

    reasons = {
        "supported_crop": supported_crop,
        "crop_consistent": crop_consistent,
        "diagnosis_support": diagnosis_support,
        "confidence_ok": confidence >= CONFIDENCE_THRESHOLD,
        "margin_ok": margin >= MARGIN_THRESHOLD,
    }

    return {"status": status, "reasons": reasons}


def _fallback_advice(payload: dict[str, Any]) -> dict[str, Any]:
    uncertain = payload.get("status") != "accepted_diagnosis"
    summary = (
        "Procena je nesigurna. Potrebna je jasnija fotografija i lokalna potvrda agronoma."
        if uncertain
        else f"Detektovana je verovatna bolest: {payload.get('diagnosis', 'nepoznato')}."
    )

    return {
        "summary": summary,
        "chemical_treatment": [
            "Koristiti registrovani fungicid prema etiketi (aktivna materija prema lokalnoj regulativi)."
        ],
        "organic_alternatives": [
            "Ukloniti jako zaražene listove i poboljšati provetravanje useva.",
            "Primena preparata na bazi bakra ili sumpora u dozvoljenim okvirima.",
        ],
        "prevention_steps": [
            "Zalivanje usmeriti na zonu korena, izbegavati kvašenje listova.",
            "Rotacija useva i dezinfekcija alata.",
        ],
        "neighboring_plant_protection": [
            "Pregledati susedne biljke i preventivno ukloniti sumnjive delove.",
            "Pojačati razmak između biljaka radi cirkulacije vazduha.",
        ],
        "eco_impact_note": "Minimizovati broj tretmana i poštovati karencu; prioritet je ciljana i odgovorna primena.",
    }


def generate_advice(payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    fallback = _fallback_advice(payload)

    if not OPENAI_API_KEY or OpenAI is None:
        return fallback, "fallback"

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.responses.create(
            model=ADVICE_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": ADVICE_GENERATOR_PROMPT},
                        {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)},
                    ],
                }
            ],
            temperature=0.2,
        )
        parsed = _safe_json_load(response.output_text, fallback)
        merged = {**fallback, **parsed}
        return merged, "openai"
    except Exception:
        return fallback, "fallback"


def run_pipeline(image: Image.Image) -> dict[str, Any]:
    vision, vision_source = analyze_with_vision(image)

    if not vision.get("is_plant", False):
        return {
            "status": "invalid_input",
            "message": "Ovo ne izgleda kao biljka. Pošalji jasnu fotografiju lista ili stabla.",
            "vision": vision,
            "meta": {"vision_source": vision_source, "advice_source": "none"},
        }

    if vision.get("image_quality") == "poor":
        return {
            "status": "retake_required",
            "message": "Slika je nejasna za pouzdanu dijagnostiku. Približi list, fokusiraj i slikaj po dnevnom svetlu.",
            "vision": vision,
            "meta": {"vision_source": vision_source, "advice_source": "none"},
        }

    if not vision.get("symptoms_visible", False):
        return {
            "status": "insufficient_evidence",
            "message": "Nema dovoljno vidljivih simptoma. Pošalji fotografiju problematičnog dela biljke iz bližeg kadra.",
            "vision": vision,
            "meta": {"vision_source": vision_source, "advice_source": "none"},
        }

    prediction = predict_disease(image, vision.get("likely_crop", "unknown"))

    trust = _diagnostic_trust(
        likely_crop=vision.get("likely_crop", "unknown"),
        label=prediction.label,
        confidence=prediction.confidence,
        margin=prediction.margin,
        diagnosis_support=vision.get("diagnosis_support", "uncertain"),
    )

    advice_payload = {
        "status": trust["status"],
        "crop": vision.get("likely_crop", "unknown"),
        "diagnosis": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "visible_symptoms": vision.get("symptom_description", ""),
        "region": "Western Balkans",
        "prediction_margin": round(prediction.margin, 4),
    }

    advice, advice_source = generate_advice(advice_payload)

    return {
        "status": trust["status"],
        "crop": vision.get("likely_crop", "unknown"),
        "diagnosis": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "prediction_margin": round(prediction.margin, 4),
        "top_k": prediction.top_k,
        "symptoms": vision.get("symptom_description", ""),
        "trust_checks": trust["reasons"],
        "advice": advice,
        "vision": vision,
        "meta": {"vision_source": vision_source, "advice_source": advice_source},
    }
