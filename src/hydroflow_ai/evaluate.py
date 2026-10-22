from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import chronological_split, read_dataset
from .metrics import event_metrics, hydrologic_metrics
from .model import ResidualCorrector


def evaluate_model(data_path: str | Path, artifacts_dir: str | Path) -> dict:
    frame = read_dataset(data_path)
    test = chronological_split(frame).test
    model = ResidualCorrector.load(Path(artifacts_dir) / "residual_corrector.joblib")
    test = test.assign(corrected_flow=model.predict(test))
    station_results: list[dict] = []
    for station, group in test.groupby("station_id", sort=True):
        baseline = hydrologic_metrics(group["observed_flow"], group["nwm_flow"])
        corrected = hydrologic_metrics(group["observed_flow"], group["corrected_flow"])
        threshold = float(group["observed_flow"].quantile(0.95))
        station_results.append(
            {
                "station_id": station,
                "baseline": baseline.as_dict(),
                "corrected": corrected.as_dict(),
                "delta_kge": corrected.kge - baseline.kge,
                "failure_cohort": bool(baseline.kge <= 0),
                "moved_above_zero": bool(baseline.kge <= 0 < corrected.kge),
                "moved_above_0_5": bool(baseline.kge <= 0 and corrected.kge >= 0.5),
                "baseline_events": event_metrics(group["observed_flow"], group["nwm_flow"], threshold),
                "corrected_events": event_metrics(group["observed_flow"], group["corrected_flow"], threshold),
            }
        )
    failures = [row for row in station_results if row["failure_cohort"]]
    good = [row for row in station_results if row["baseline"]["kge"] > 0.5]
    summary = {
        "stations": len(station_results),
        "negative_kge_stations": len(failures),
        "negative_kge_moved_above_zero": sum(row["moved_above_zero"] for row in failures),
        "negative_kge_moved_above_0_5": sum(row["moved_above_0_5"] for row in failures),
        "median_delta_kge": float(np.median([row["delta_kge"] for row in station_results])),
        "good_stations_degraded": sum(row["delta_kge"] < -0.05 for row in good),
    }
    result = {"summary": summary, "stations": station_results}
    output = Path(artifacts_dir) / "evaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    test.to_csv(Path(artifacts_dir) / "test_predictions.csv", index=False)
    return result

