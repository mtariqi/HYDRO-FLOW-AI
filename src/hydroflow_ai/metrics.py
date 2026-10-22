from __future__ import annotations

import numpy as np

from .schemas import MetricBundle


EPS = 1e-9


def _arrays(observed, predicted) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    keep = np.isfinite(obs) & np.isfinite(pred)
    if keep.sum() < 2:
        raise ValueError("At least two finite observation/prediction pairs are required")
    return obs[keep], pred[keep]


def hydrologic_metrics(observed, predicted) -> MetricBundle:
    obs, pred = _arrays(observed, predicted)
    obs_mean = float(obs.mean())
    pred_mean = float(pred.mean())
    obs_std = float(obs.std(ddof=0))
    pred_std = float(pred.std(ddof=0))
    correlation = float(np.corrcoef(obs, pred)[0, 1]) if obs_std > EPS and pred_std > EPS else 0.0
    variability_ratio = pred_std / max(obs_std, EPS)
    bias_ratio = pred_mean / max(abs(obs_mean), EPS)
    kge = 1.0 - np.sqrt(
        (correlation - 1.0) ** 2
        + (variability_ratio - 1.0) ** 2
        + (bias_ratio - 1.0) ** 2
    )
    residual = pred - obs
    denominator = float(np.sum((obs - obs_mean) ** 2))
    nse = 1.0 - float(np.sum(residual**2)) / max(denominator, EPS)
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    percent_bias = 100.0 * float(np.sum(residual)) / max(abs(float(np.sum(obs))), EPS)
    return MetricBundle(
        kge=float(kge),
        nse=nse,
        rmse=rmse,
        mae=mae,
        percent_bias=percent_bias,
        correlation=correlation,
        variability_ratio=float(variability_ratio),
        bias_ratio=float(bias_ratio),
    )


def event_metrics(observed, predicted, threshold: float) -> dict[str, float]:
    obs, pred = _arrays(observed, predicted)
    observed_event = obs >= threshold
    predicted_event = pred >= threshold
    hits = int(np.sum(observed_event & predicted_event))
    misses = int(np.sum(observed_event & ~predicted_event))
    false_alarms = int(np.sum(~observed_event & predicted_event))
    csi = hits / max(hits + misses + false_alarms, 1)
    detection = hits / max(hits + misses, 1)
    false_alarm_ratio = false_alarms / max(hits + false_alarms, 1)
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "csi": csi,
        "probability_of_detection": detection,
        "false_alarm_ratio": false_alarm_ratio,
    }

