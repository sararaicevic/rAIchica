"""Central configuration for rAIchica demo."""

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")
ADVICE_MODEL = os.getenv("ADVICE_MODEL", "gpt-4o-mini")

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
