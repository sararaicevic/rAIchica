"""Core pipeline logic for the rAIchica Streamlit demo."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
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
from .prompts import VISION_GATEKEEPER_PROMPT

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - import safety for environments without package
    OpenAI = None

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover - import safety for environments without package
    load_model = None

logger = logging.getLogger(__name__)


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
        logger.info("Local Keras loader is unavailable; falling back to stub predictions.")
        return None

    path = Path(MODEL_PATH)
    if not path.exists():
        logger.warning("Local model file does not exist: %s", path)
        return None

    try:
        logger.info("Loading local .h5 model from %s", path)
        return load_model(path, compile=False)
    except Exception:
        logger.exception("Failed to load local .h5 model from %s", path)
        return None


def get_local_model_status() -> dict[str, Any]:
    path = Path(MODEL_PATH)
    class_names_path = Path(CLASS_NAMES_PATH)

    if load_model is None:
        return {
            "available": False,
            "reason": "tensorflow_unavailable",
            "model_path": str(path),
            "model_exists": path.exists(),
            "class_names_path": str(class_names_path),
            "class_names_exists": class_names_path.exists(),
        }

    model = _load_local_model()
    if model is None:
        return {
            "available": False,
            "reason": "model_load_failed_or_missing_file",
            "model_path": str(path),
            "model_exists": path.exists(),
            "class_names_path": str(class_names_path),
            "class_names_exists": class_names_path.exists(),
        }

    return {
        "available": True,
        "reason": "loaded",
        "model_path": str(path),
        "model_exists": path.exists(),
        "class_names_path": str(class_names_path),
        "class_names_exists": class_names_path.exists(),
    }


def _predict_with_local_h5(image: Image.Image) -> tuple[LocalPrediction | None, dict[str, Any]]:
    model = _load_local_model()
    if model is None:
        return None, {"source": "local_h5", "available": False}

    class_names = _load_class_names()
    resized = image.copy().convert("RGB").resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
    arr = np.array(resized, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    try:
        preds = model.predict(arr, verbose=0)
    except Exception:
        logger.exception("Local model prediction failed")
        return None, {"source": "local_h5", "available": True, "error": "prediction_failed"}

    if preds is None or len(preds) == 0:
        logger.warning("Local model returned an empty prediction payload")
        return None, {"source": "local_h5", "available": True, "error": "empty_prediction"}

    raw_output = np.array(preds[0], dtype=np.float32)
    logger.info("Local model raw output: %s", raw_output.tolist())

    probs = np.array(raw_output, dtype=np.float32)
    if probs.ndim != 1 or probs.size == 0:
        logger.warning("Local model output has unexpected shape: %s", probs.shape)
        return None, {
            "source": "local_h5",
            "available": True,
            "error": "invalid_shape",
            "raw_output": raw_output.tolist(),
        }

    probs = np.maximum(probs, 0)
    if probs.sum() <= 0:
        logger.warning("Local model output sums to zero after sanitization")
        return None, {
            "source": "local_h5",
            "available": True,
            "error": "non_positive_output",
            "raw_output": raw_output.tolist(),
        }
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

    prediction = LocalPrediction(
        label=class_names[top_idx],
        confidence=float(probs[top_idx]),
        top_k=top_k,
        margin=float(probs[top_idx] - probs[second_idx]),
    )
    debug = {
        "source": "local_h5",
        "available": True,
        "model_path": str(Path(MODEL_PATH)),
        "input_size": MODEL_INPUT_SIZE,
        "class_names": class_names,
        "raw_output": raw_output.tolist(),
        "normalized_probs": probs.tolist(),
        "predicted_label": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "margin": round(prediction.margin, 4),
        "top_k": top_k,
    }
    logger.info(
        "Local model prediction: label=%s confidence=%.4f margin=%.4f",
        prediction.label,
        prediction.confidence,
        prediction.margin,
    )
    return prediction, debug


def _fallback_vision_analysis(image: Image.Image) -> dict[str, Any]:
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)

    brightness = float(arr.mean())
    sharpness = float(np.var(np.diff(arr, axis=0))) if arr.shape[0] > 1 else 0.0
    green_ratio = float((arr[:, :, 1] > arr[:, :, 0]).mean())
    skin_like = (
        (arr[:, :, 0] > 95)
        & (arr[:, :, 1] > 40)
        & (arr[:, :, 2] > 20)
        & (arr[:, :, 0] > arr[:, :, 1])
        & (arr[:, :, 1] > arr[:, :, 2])
    )
    skin_ratio = float(skin_like.mean())

    contains_person = skin_ratio > 0.20
    is_plant = green_ratio > 0.30
    quality = "good" if brightness > 45 and sharpness > 30 else "poor"
    symptoms_visible = bool(((arr[:, :, 0] > 150) & (arr[:, :, 1] < 120)).mean() > 0.05)

    likely_crop = "tomato" if is_plant else "unknown"
    primary_subject = "plant" if is_plant else ("human" if contains_person else "other")
    return {
        "is_plant": is_plant,
        "contains_person": contains_person,
        "primary_subject": primary_subject,
        "image_quality": quality,
        "likely_crop": likely_crop,
        "symptoms_visible": symptoms_visible,
        "symptom_description": (
            "Possible leaf discoloration visible." if symptoms_visible else "No clear disease symptoms visible."
        ),
        "rejection_reason": "Human-like skin tones dominate the image." if contains_person else "",
        "diagnosis_support": "supported" if likely_crop in SUPPORTED_CROPS else "uncertain",
    }


def run_vision_gatekeeper(image: Image.Image) -> tuple[dict[str, Any], str]:
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
        primary_subject = str(merged.get("primary_subject", "")).strip().lower()
        contains_person = bool(merged.get("contains_person", False)) or primary_subject == "human"

        if primary_subject == "plant":
            merged["contains_person"] = False
            merged["is_plant"] = True
            merged["primary_subject"] = "plant"
            merged["rejection_reason"] = ""
        elif primary_subject == "human":
            merged["contains_person"] = True
            merged["is_plant"] = False
            merged["primary_subject"] = "human"
            merged["likely_crop"] = "unknown"
            merged["symptoms_visible"] = False
            merged["rejection_reason"] = merged.get("rejection_reason") or "Human-centric scene detected."
        elif contains_person and primary_subject not in {"plant", "other"}:
            merged["contains_person"] = True
            merged["primary_subject"] = primary_subject or "human"
            merged["is_plant"] = False
            merged["likely_crop"] = "unknown"
            merged["symptoms_visible"] = False
            merged["rejection_reason"] = merged.get("rejection_reason") or "Human-centric or non-plant scene detected."
        elif not merged.get("is_plant", False):
            merged["likely_crop"] = "unknown"

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


def _fallback_stub_prediction(image: Image.Image, likely_crop: str) -> LocalPrediction:
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


def predict_with_local_model(image: Image.Image, top_k: int = TOP_K, likely_crop: str = "unknown") -> dict[str, Any]:
    local_prediction, _ = _predict_with_local_h5(image)
    if local_prediction is not None:
        results = local_prediction.top_k[:top_k]
        return {
            "model_source": "local_h5",
            "status": "completed",
            "predicted_class": local_prediction.label,
            "confidence": round(local_prediction.confidence, 4),
            "margin": round(local_prediction.margin, 4),
            "top_predictions": [
                {"class_name": item["label"], "confidence": float(item["probability"])}
                for item in results
            ],
        }

    fallback_prediction = _fallback_stub_prediction(image, likely_crop)
    return {
        "model_source": "fallback_stub",
        "status": "completed_with_fallback",
        "predicted_class": fallback_prediction.label,
        "confidence": round(fallback_prediction.confidence, 4),
        "margin": round(fallback_prediction.margin, 4),
        "top_predictions": [
            {"class_name": item["label"], "confidence": float(item["probability"])}
            for item in fallback_prediction.top_k[:top_k]
        ],
    }


def _decision_from_values(
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

    return {
        "status": status,
        "reasons": reasons,
        "accepted": accepted,
        "message": "Diagnosis accepted." if accepted else "Diagnosis is uncertain and should be treated as a lower-confidence output.",
    }


def decision_layer(vision_result: dict[str, Any], model_result: dict[str, Any]) -> dict[str, Any]:
    predicted_class = model_result.get("predicted_class", "unknown")
    confidence = float(model_result.get("confidence", 0.0))
    margin = float(model_result.get("margin", 0.0))
    likely_crop = vision_result.get("likely_crop", "unknown")
    diagnosis_support = vision_result.get("diagnosis_support", "uncertain")
    result = _decision_from_values(likely_crop, predicted_class, confidence, margin, diagnosis_support)
    result["inputs"] = {
        "likely_crop": likely_crop,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "diagnosis_support": diagnosis_support,
    }
    return result


def _diagnostic_trust(
    likely_crop: str,
    label: str,
    confidence: float,
    margin: float,
    diagnosis_support: str,
) -> dict[str, Any]:
    return _decision_from_values(likely_crop, label, confidence, margin, diagnosis_support)


def _fallback_advice(payload: dict[str, Any]) -> dict[str, Any]:
    uncertain = payload.get("status") != "accepted_diagnosis"
    summary = (
        "The assessment is uncertain. A clearer photo and local agronomist confirmation are needed."
        if uncertain
        else f"A likely disease was detected: {payload.get('diagnosis', 'unknown')}."
    )

    return {
        "summary": summary,
        "chemical_treatment": [
            "Use a registered fungicide according to the label and local regulations."
        ],
        "organic_alternatives": [
            "Remove heavily infected leaves and improve crop airflow.",
            "Apply copper- or sulfur-based products only where permitted.",
        ],
        "prevention_steps": [
            "Water at the root zone and avoid wetting the leaves.",
            "Rotate crops and disinfect tools.",
        ],
        "neighboring_plant_protection": [
            "Inspect neighboring plants and remove suspicious parts early.",
            "Increase spacing between plants to improve air circulation.",
        ],
        "eco_impact_note": "Minimize treatments and respect pre-harvest intervals; prioritize targeted, responsible application.",
    }


def _trace_entry(step: str, status: str, returned: dict[str, Any] | None = None, note: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"step": step, "status": status}
    if returned is not None:
        entry["returned"] = returned
    if note is not None:
        entry["note"] = note
    return entry


def _json_schema_farmer_advice() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "chemical_treatment": {"type": "array", "items": {"type": "string"}},
            "organic_alternatives": {"type": "array", "items": {"type": "string"}},
            "prevention_steps": {"type": "array", "items": {"type": "string"}},
            "neighboring_plant_protection": {"type": "array", "items": {"type": "string"}},
            "eco_impact_note": {"type": "string"},
        },
        "required": [
            "summary",
            "chemical_treatment",
            "organic_alternatives",
            "prevention_steps",
            "neighboring_plant_protection",
            "eco_impact_note",
        ],
        "additionalProperties": False,
    }


def generate_farmer_advice(
    vision_result: dict[str, Any],
    model_result: dict[str, Any],
    decision_result: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "region": "Western Balkans",
        "vision_result": vision_result,
        "model_result": model_result,
        "decision_result": decision_result,
    }

    fallback = _fallback_advice(
        {
            "status": decision_result.get("status", "uncertain_diagnosis"),
            "diagnosis": model_result.get("predicted_class", "unknown"),
        }
    )

    if not OPENAI_API_KEY or OpenAI is None:
        return fallback

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.responses.create(
            model=ADVICE_MODEL,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a practical agronomy advisor for farmers in the Western Balkans. "
                                "Be concise, practical, and uncertainty-aware. "
                                "If diagnosis is uncertain, say so clearly and avoid strong claims."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Using the JSON below, generate farmer guidance.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2),
                        }
                    ],
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "farmer_advice",
                    "schema": _json_schema_farmer_advice(),
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text)
    except Exception:
        return fallback


def run_full_pipeline(image: Image.Image) -> dict[str, Any]:
    model_loader_status = get_local_model_status()
    vision_result, vision_source = run_vision_gatekeeper(image)
    vision_result = {**vision_result, "source": vision_source}

    if vision_result.get("contains_person", False):
        decision_result = {
            "status": "reject",
            "message": "This image contains a person, not a plant.",
        }
        return {
            "vision_result": vision_result,
            "model_result": None,
            "decision_result": decision_result,
            "advice_result": None,
            "trace": [
                _trace_entry("vision_gatekeeper", "completed", vision_result),
                _trace_entry("input_triage", "stopped", decision_result, vision_result.get("rejection_reason", "")),
            ],
            "status": "invalid_input",
            "message": "This image contains a person, not a plant. Please upload a clear photo of a leaf or plant.",
            "vision": vision_result,
            "meta": {"vision_source": vision_source, "advice_source": "none"},
        }

    if not vision_result.get("is_plant", False):
        decision_result = {
            "status": "reject",
            "message": "This does not appear to be a plant.",
        }
        return {
            "vision_result": vision_result,
            "model_result": None,
            "decision_result": decision_result,
            "advice_result": None,
            "trace": [
                _trace_entry("vision_gatekeeper", "completed", vision_result),
                _trace_entry("input_triage", "stopped", decision_result, "The image does not look like a plant."),
            ],
            "status": "invalid_input",
            "message": "This does not appear to be a plant. Please upload a clear photo of a leaf or plant.",
            "vision": vision_result,
            "meta": {"vision_source": vision_source, "advice_source": "none"},
        }

    uncertain = vision_result.get("image_quality") != "good" or not vision_result.get("symptoms_visible", False)

    model_result = predict_with_local_model(image, top_k=TOP_K, likely_crop=vision_result.get("likely_crop", "unknown"))
    decision_result = decision_layer(vision_result, model_result)
    if uncertain:
        decision_result = {
            **decision_result,
            "status": "uncertain_diagnosis",
            "message": "The image is usable, but confidence is limited because image quality or symptom visibility is poor.",
            "uncertain": True,
        }
    advice_result = generate_farmer_advice(vision_result, model_result, decision_result)

    trace = [
        _trace_entry("vision_gatekeeper", "completed", vision_result),
        _trace_entry(
            "input_triage",
            "completed" if not uncertain else "completed_uncertain",
            {
                "status": decision_result["status"],
                "message": decision_result["message"],
                "uncertain": uncertain,
            },
            "Image quality or symptom visibility is insufficient." if uncertain else "Image is usable for diagnosis.",
        ),
        _trace_entry("local_model_loader", "completed", model_loader_status),
        _trace_entry("local_h5_model", "completed" if model_result["model_source"] == "local_h5" else "completed_with_fallback", model_result),
        _trace_entry("decision_layer", "completed", decision_result),
        _trace_entry("advice_generator", "completed", {"summary": advice_result.get("summary", ""), "source": "openai" if OPENAI_API_KEY and OpenAI is not None else "fallback"}),
    ]

    return {
        "vision_result": vision_result,
        "model_result": model_result,
        "decision_result": decision_result,
        "advice_result": advice_result,
        "trace": trace,
        "status": decision_result["status"],
        "message": decision_result.get("message", ""),
        "crop": vision_result.get("likely_crop", "unknown"),
        "diagnosis": model_result.get("predicted_class", ""),
        "confidence": round(float(model_result.get("confidence", 0.0)), 4),
        "prediction_margin": round(float(model_result.get("margin", 0.0)), 4),
        "top_k": [
            {"label": item["class_name"], "probability": round(float(item["confidence"]), 4)}
            for item in model_result.get("top_predictions", [])
        ],
        "symptoms": vision_result.get("symptom_description", ""),
        "trust_checks": decision_result.get("reasons", {}),
        "vision": vision_result,
        "model_loader_status": model_loader_status,
        "meta": {"vision_source": vision_source, "advice_source": "openai" if OPENAI_API_KEY and OpenAI is not None else "fallback"},
    }


def run_pipeline(image: Image.Image) -> dict[str, Any]:
    return run_full_pipeline(image)
