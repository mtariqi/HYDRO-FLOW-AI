"""Regression tests for screen_scale_anomalies.py.

HIDDEN_BY_OUTLIER is the most important case here: it's not a syntax
check, it demonstrates the actual reason the central-quantile overlap
diagnostic exists (a naive full min/max overlap check is fooled by a
single outlier in each series; the central-range check is not).
"""
import numpy as np
import pandas as pd
import pytest

from hydro_flow_ai.screen_scale_anomalies import (
    ScaleAuditError,
    screen_for_scale_anomalies,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_station(rng, station_id, obs, nwm):
    return pd.DataFrame({"station_id": station_id, "flow_cms": obs, "NWM_flow": nwm})


def test_normal_station_not_flagged(rng):
    obs = rng.gamma(2, 5, 1000)
    nwm = obs * 1.1 + rng.normal(0, 1, 1000)
    df = make_station(rng, "NORMAL", obs, nwm)
    result = screen_for_scale_anomalies(df)
    assert not result.loc[result.station_id == "NORMAL", "flagged"].iloc[0]


def test_regulated_like_station_flagged_both_ways(rng):
    obs = rng.uniform(0.1, 9, 1000)
    nwm = rng.uniform(88, 228, 1000)
    df = make_station(rng, "REGULATED_LIKE", obs, nwm)
    result = screen_for_scale_anomalies(df).set_index("station_id")
    assert result.loc["REGULATED_LIKE", "no_range_overlap"]
    assert result.loc["REGULATED_LIKE", "large_scale_ratio"]
    assert result.loc["REGULATED_LIKE", "flagged"]


def test_hidden_by_outlier_caught_by_central_range_not_full_range(rng):
    """Bulk distributions are totally separate, but one planted outlier
    in each series makes the raw min/max ranges technically overlap.
    The full-range check must MISS this; the central-range check must
    CATCH it anyway."""
    n = 1000
    obs = rng.uniform(1, 5, n)
    nwm = rng.uniform(50, 100, n)
    obs[0] = 80.0
    nwm[0] = 3.0
    df = make_station(rng, "HIDDEN_BY_OUTLIER", obs, nwm)
    result = screen_for_scale_anomalies(df).set_index("station_id")

    assert not result.loc["HIDDEN_BY_OUTLIER", "no_range_overlap"], (
        "full range check should be fooled by the planted outliers"
    )
    assert result.loc["HIDDEN_BY_OUTLIER", "no_central_range_overlap"], (
        "central-quantile check must catch what the full-range check misses"
    )
    assert result.loc["HIDDEN_BY_OUTLIER", "flagged"]


def test_custom_quantile_bounds_reflected_in_output_columns(rng):
    obs = rng.gamma(2, 5, 1000)
    nwm = obs * 1.1
    df = make_station(rng, "NORMAL", obs, nwm)
    result = screen_for_scale_anomalies(df, lower_quantile=0.10, upper_quantile=0.90)
    expected_q_low = float(np.quantile(obs, 0.10))
    actual_q_low = result.loc[result.station_id == "NORMAL", "obs_q_low_cms"].iloc[0]
    assert actual_q_low == pytest.approx(expected_q_low)


def test_missing_required_column_raises_clear_error():
    df = pd.DataFrame({"station_id": ["A"], "flow_cms": [1.0]})  # missing NWM_flow
    with pytest.raises(ScaleAuditError, match="NWM_flow"):
        screen_for_scale_anomalies(df)


def test_station_with_insufficient_rows_is_skipped_not_crashed(rng):
    obs_ok = rng.gamma(2, 5, 100)
    nwm_ok = obs_ok * 1.1
    df_ok = make_station(rng, "ENOUGH_DATA", obs_ok, nwm_ok)

    df_tiny = make_station(rng, "TOO_FEW_ROWS", [1.0, 2.0], [1.1, 2.1])

    df = pd.concat([df_ok, df_tiny], ignore_index=True)
    result = screen_for_scale_anomalies(df, min_rows=20)
    assert "TOO_FEW_ROWS" not in result["station_id"].values
    assert "ENOUGH_DATA" in result["station_id"].values


def test_ratio_threshold_must_exceed_one(rng):
    obs = rng.gamma(2, 5, 100)
    df = make_station(rng, "A", obs, obs)
    with pytest.raises(ValueError, match="ratio_threshold"):
        screen_for_scale_anomalies(df, ratio_threshold=1.0)


def test_quantile_bounds_must_be_ordered(rng):
    obs = rng.gamma(2, 5, 100)
    df = make_station(rng, "A", obs, obs)
    with pytest.raises(ValueError, match="Quantiles"):
        screen_for_scale_anomalies(df, lower_quantile=0.9, upper_quantile=0.1)
