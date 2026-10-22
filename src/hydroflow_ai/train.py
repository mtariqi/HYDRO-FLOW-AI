from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import chronological_split, read_dataset
from .metrics import hydrologic_metrics
from .model import ResidualCorrector


def train_model(data_path: str | Path, output_dir: str | Path) -> dict:
    frame = read_dataset(data_path)
    split = chronological_split(frame)
    model = ResidualCorrector.create().fit(split.train)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "residual_corrector.joblib"
    model.save(model_path)

    baseline = hydrologic_metrics(split.validation["observed_flow"], split.validation["nwm_flow"])
    corrected = hydrologic_metrics(split.validation["observed_flow"], model.predict(split.validation))
    metadata = {
        "model_type": "physics_guided_residual_hist_gradient_boosting",
        "features": model.features,
        "validation": {
            "baseline": baseline.as_dict(),
            "corrected": corrected.as_dict(),
            "delta_kge": corrected.kge - baseline.kge,
        },
        "rows": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
    }
    (output / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata

