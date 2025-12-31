import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from src.models.predict import load_model, predict_one
from .schemas import PredictRequest, PredictResponse


APP_NAME = "fraud-mlops-api"
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "models/metrics.json")
REFERENCE_STATS_PATH = os.getenv("REFERENCE_STATS_PATH", "models/reference_stats.json")

app = FastAPI(title=APP_NAME)

_model_obj: Dict[str, Any] | None = None


def _read_json_if_exists(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@app.on_event("startup")
def _startup():
    global _model_obj
    try:
        _model_obj = load_model(MODEL_PATH)
    except Exception as e:
        _model_obj = None
        print(f"Model load failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_obj is not None, "model_path": MODEL_PATH}


@app.get("/metrics")
def metrics():
    return _read_json_if_exists(METRICS_PATH)


@app.get("/reference-stats")
def reference_stats():
    return _read_json_if_exists(REFERENCE_STATS_PATH)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model_obj is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH / training artifacts.")
    try:
        res = predict_one(req.features, _model_obj)
        return PredictResponse(request_id=req.request_id, **res)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
