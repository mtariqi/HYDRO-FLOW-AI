from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .data import FEATURES
from .diagnostics import diagnose_station
from .model import ResidualCorrector


ARTIFACTS = Path(os.getenv("HYDROFLOW_ARTIFACTS", "artifacts"))
app = FastAPI(title="HYDRO-FLOW-AI", version="0.2.0")


class ForecastRequest(BaseModel):
    station_id: str
    features: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_available": (ARTIFACTS / "residual_corrector.joblib").exists()}


@app.post("/predict")
def predict(request: ForecastRequest):
    missing = sorted(set(FEATURES).difference(request.features))
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    model = ResidualCorrector.load(ARTIFACTS / "residual_corrector.joblib")
    frame = pd.DataFrame([{name: request.features[name] for name in FEATURES}])
    return {
        "station_id": request.station_id,
        "corrected_flow": float(model.predict(frame)[0]),
        "units": "synthetic_demo_units",
        "warning": "Research output; not an official forecast or warning.",
    }


@app.get("/diagnose/{station_id}")
def diagnose(station_id: str):
    try:
        return {"station_id": station_id, "report": diagnose_station(ARTIFACTS, station_id)}
    except FileNotFoundError as error:
        raise HTTPException(404, str(error)) from error

