"""Central configuration for rAIchica demo."""

import os
from pathlib import Path

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
ADVICE_MODEL = os.getenv("ADVICE_MODEL", "gpt-4o-mini")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "raichica_v1.h5"))
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", str(PROJECT_ROOT / "models" / "class_names.txt"))
CLASS_NAMES_CSV = os.getenv("CLASS_NAMES", "")
MODEL_INPUT_SIZE = int(os.getenv("MODEL_INPUT_SIZE", "224"))

SUPPORTED_CROPS = {
    "tomato",
    "potato",
    "pepper",
    "grape",
    "apple",
    "corn",
    "wheat",
    "strawberry",
    "cucumber",
}

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
MARGIN_THRESHOLD = float(os.getenv("MARGIN_THRESHOLD", "0.15"))

TOP_K = 3
MAX_IMAGE_SIZE = 1600
