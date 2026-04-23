# rAIchica

Streamlit demo for a hybrid agricultural diagnosis pipeline:
- Vision gatekeeper validates image and extracts crop/symptom context.
- Local disease model predicts label, confidence, and prediction margin.
- Trust policy decides accepted vs uncertain diagnosis.
- Advice generator returns farmer-friendly recommendations for the Western Balkans.

## Architecture

```text
User Photo
   ↓
[Streamlit Frontend]
   ↓
[Vision Gatekeeper - GPT-4o Vision]
   ↓
[Decision Layer 1: Input Triage]
   ↓
[Local Disease Model - MobileNetV2 (demo stub)]
   ↓
[Decision Layer 2: Diagnostic Trust Policy]
   ↓
[Advice Generator - GPT-4o Mini]
   ↓
[Final Farmer-Friendly Result]
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export $(grep -v '^#' .env | xargs)
streamlit run app.py
```

Open `http://localhost:8501`.

## Environment variables

- `OPENAI_API_KEY` - optional; if missing, app runs fallback mode.
- `VISION_MODEL` - default `gpt-4o`
- `ADVICE_MODEL` - default `gpt-4o-mini`
- `CONFIDENCE_THRESHOLD` - default `0.80`
- `MARGIN_THRESHOLD` - default `0.15`

## Streamlit Cloud deploy

1. Push repository to GitHub.
2. Create app in Streamlit Community Cloud.
3. Set main file to `app.py`.
4. Add secret `OPENAI_API_KEY` in app settings.
5. Deploy.

## Notes

- The local classifier in this demo is a deterministic stub for presentation.
- Replace `predict_disease` in `src/pipeline.py` with your real MobileNetV2 inference for production.
