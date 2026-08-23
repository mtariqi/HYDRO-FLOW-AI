"""Tests for the exclusion registry and station QC summary join.

The critical behavior under test: a station flagged by the scale
screen but ABSENT from the registry must default to model_eligible=False
with a loud warning -- never silently to True. Detection and exclusion
are separate decisions, but an unadjudicated flag must never look the
same as "reviewed and safe."
"""
from pathlib import Path

import pandas as pd
import pytest

from hydro_flow_ai.qc_registry import (
    RegistryError,
    build_station_qc_summary,
    load_exclusion_registry,
)


@pytest.fixture
def station_metrics():
    return pd.DataFrame({
        "station_id": ["A", "B", "C"],
        "station_rows": [1000, 1000, 1000],
        "overall_rmse_cms": [2.0, 5.0, 50.0],
        "overall_bias_cms": [0.5, 1.0, 40.0],
        "q95_rmse_cms": [3.0, 8.0, 60.0],
        "q99_rmse_cms": [4.0, 10.0, 70.0],
    })


@pytest.fixture
def scale_screen():
    return pd.DataFrame({
        "station_id": ["A", "B", "C"],
        "mean_scale_ratio": [1.1, 1.3, 30.0],
        "median_scale_ratio": [1.1, 1.3, 28.0],
        "central_overlap_fraction": [0.9, 0.8, 0.0],
        "flagged": [False, False, True],
    })


def test_unflagged_unregistered_stations_default_include(station_metrics, scale_screen):
    empty_registry = load_exclusion_registry(None)
    summary = build_station_qc_summary(station_metrics, scale_screen, empty_registry).set_index("station_id")
    assert summary.loc["A", "model_eligible"]
    assert summary.loc["B", "model_eligible"]


def test_flagged_unregistered_station_defaults_excluded_with_warning(station_metrics, scale_screen, caplog):
    empty_registry = load_exclusion_registry(None)
    with caplog.at_level("WARNING"):
        summary = build_station_qc_summary(station_metrics, scale_screen, empty_registry).set_index("station_id")
    assert not summary.loc["C", "model_eligible"]
    assert summary.loc["C", "registry_status"] == "review"
    assert any("NO exclusion-registry entry" in message for message in caplog.messages)


def test_registered_review_status_stays_excluded(station_metrics, scale_screen):
    registry = pd.DataFrame({
        "station_id": ["C"], "status": ["review"], "reason": ["under investigation"],
        "evidence": ["scale screen"], "decision_source": ["hydrologic_qc"],
    })
    summary = build_station_qc_summary(station_metrics, scale_screen, registry).set_index("station_id")
    assert not summary.loc["C", "model_eligible"]


def test_registered_include_status_overrides_flag(station_metrics, scale_screen):
    registry = pd.DataFrame({
        "station_id": ["C"], "status": ["include"], "reason": ["verified false positive"],
        "evidence": ["crosswalk check"], "decision_source": ["hydrologic_qc"],
    })
    summary = build_station_qc_summary(station_metrics, scale_screen, registry).set_index("station_id")
    assert summary.loc["C", "model_eligible"]


def test_invalid_status_value_raises(tmp_path):
    bad_path = tmp_path / "bad_registry.csv"
    pd.DataFrame({
        "station_id": ["C"], "status": ["maybe"], "reason": ["x"],
        "evidence": ["x"], "decision_source": ["x"],
    }).to_csv(bad_path, index=False)

    with pytest.raises(RegistryError, match="invalid status"):
        load_exclusion_registry(bad_path)


def test_missing_registry_file_returns_empty_not_error(tmp_path):
    registry = load_exclusion_registry(tmp_path / "does_not_exist.csv")
    assert registry.empty
    assert list(registry.columns) == ["station_id", "status", "reason", "evidence", "decision_source"]
