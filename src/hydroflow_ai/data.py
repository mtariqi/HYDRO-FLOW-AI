from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import Split


FEATURES = [
    "nwm_flow",
    "precipitation",
    "temperature",
    "soil_moisture",
    "swe",
    "reservoir_storage_change",
    "recent_residual",
    "upstream_flow",
    "basin_area",
    "slope",
    "regulated",
    "sin_doy",
    "cos_doy",
]


def make_synthetic_dataset(
    stations: int = 12,
    periods: int = 1600,
    frequency: str = "15min",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a connected, regime-diverse river network for smoke tests."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2021-01-01", periods=periods, freq=frequency, tz="UTC")
    frames: list[pd.DataFrame] = []
    upstream_truth = np.zeros(periods)

    for idx in range(stations):
        station = f"S{idx:03d}"
        regulated = int(idx % 4 == 0)
        basin_area = float(120 + 45 * idx + rng.uniform(0, 30))
        slope = float(max(0.001, 0.08 - idx * 0.004))
        day = timestamps.dayofyear.to_numpy()
        temperature = 8 + 14 * np.sin(2 * np.pi * (day - 90) / 365) + rng.normal(0, 3, periods)
        storm = rng.gamma(1.2, 2.0, periods) * (rng.random(periods) < 0.10)
        precipitation = np.convolve(storm, np.array([0.55, 0.30, 0.15]), mode="same")
        swe = np.maximum(0, 90 - temperature * 4 + rng.normal(0, 6, periods))
        soil = np.clip(0.25 + np.convolve(precipitation, np.ones(16) / 45, mode="same"), 0, 1)
        storage_change = (
            np.convolve(precipitation, np.ones(12) / 30, mode="same")
            - np.convolve(np.maximum(temperature, 0), np.ones(12) / 300, mode="same")
        ) if regulated else np.zeros(periods)
        seasonal = 5 + 2.5 * np.sin(2 * np.pi * day / 365)
        runoff = precipitation * (1.5 + 2.5 * soil) + np.maximum(temperature, 0) * swe / 500
        observed = np.maximum(
            0.05,
            seasonal + runoff + 0.20 * upstream_truth - regulated * 1.8 * storage_change
            + rng.normal(0, 0.65, periods),
        )
        if idx % 5 == 0:
            # Deliberately poor timing and amplitude emulate the failure cohort
            # visible in nationwide station-skill maps.
            nwm_flow = np.maximum(
                0.05,
                0.30 * np.roll(observed, 48)
                + 3.5
                + 1.2 * np.roll(precipitation, 12)
                + rng.normal(0, 1.8, periods),
            )
        else:
            regional_bias = 1.25 if idx % 5 == 1 else 0.9
            nwm_flow = np.maximum(
                0.05,
                observed * regional_bias
                + 0.8 * np.roll(precipitation, 3)
                + regulated * 2.2 * storage_change
                + rng.normal(0, 1.1, periods),
            )
        residual = observed - nwm_flow
        recent_residual = pd.Series(residual).shift(1).rolling(4, min_periods=1).mean().fillna(0).to_numpy()
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "station_id": station,
                "downstream_station_id": f"S{idx + 1:03d}" if idx < stations - 1 else "OUTLET",
                "observed_flow": observed,
                "nwm_flow": nwm_flow,
                "precipitation": precipitation,
                "temperature": temperature,
                "soil_moisture": soil,
                "swe": swe,
                "reservoir_storage_change": storage_change,
                "recent_residual": recent_residual,
                "upstream_flow": upstream_truth,
                "basin_area": basin_area,
                "slope": slope,
                "regulated": regulated,
                "sin_doy": np.sin(2 * np.pi * day / 365),
                "cos_doy": np.cos(2 * np.pi * day / 365),
            }
        )
        frames.append(frame)
        upstream_truth = observed
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "station_id"])


def validate_dataset(frame: pd.DataFrame) -> None:
    required = {"timestamp", "station_id", "observed_flow", *FEATURES}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.duplicated(["timestamp", "station_id"]).any():
        raise ValueError("Duplicate timestamp/station rows detected")
    if (frame["observed_flow"] < 0).any() or (frame["nwm_flow"] < 0).any():
        raise ValueError("Flow values must be non-negative")
    if not np.isfinite(frame[["observed_flow", *FEATURES]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite modelling values detected")


def chronological_split(frame: pd.DataFrame, train_fraction=0.70, validation_fraction=0.15) -> Split:
    timestamps = np.array(sorted(pd.to_datetime(frame["timestamp"], utc=True).unique()))
    first = int(len(timestamps) * train_fraction)
    second = int(len(timestamps) * (train_fraction + validation_fraction))
    if first < 1 or second <= first or second >= len(timestamps):
        raise ValueError("Split fractions do not leave non-empty train/validation/test periods")
    parsed = pd.to_datetime(frame["timestamp"], utc=True)
    return Split(
        train=frame.loc[parsed < timestamps[first]].copy(),
        validation=frame.loc[(parsed >= timestamps[first]) & (parsed < timestamps[second])].copy(),
        test=frame.loc[parsed >= timestamps[second]].copy(),
    )


def read_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    validate_dataset(frame)
    return frame

