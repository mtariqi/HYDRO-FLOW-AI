"""Tests for the fail-closed feature contract scaffold.

Covers:
- approved-feature selection
- explicit approved requests
- rejection of review/exclude/target features
- unregistered dataset columns
- stale registry entries
- duplicate registry entries
- invalid registry statuses
- duplicate requested features
- missing required registry columns
- missing/null registry statuses
- missing/null registry feature names
"""
import pandas as pd
import pytest

from hydro_flow_ai.feature_contract import FeatureContractError, enforce_feature_contract


def make_registry():
    return pd.DataFrame({
        "feature_name": ["NWM_flow", "temperature", "station_id", "flow_cms"],
        "status": ["include", "review", "exclude", "target"],
    })


def test_returns_only_approved_features():
    result = enforce_feature_contract({"NWM_flow", "temperature", "station_id", "flow_cms"}, make_registry())
    assert result == ["NWM_flow"]


def test_explicit_approved_request_passes():
    result = enforce_feature_contract(
        {"NWM_flow", "temperature", "station_id", "flow_cms"}, make_registry(), requested_features=["NWM_flow"]
    )
    assert result == ["NWM_flow"]


@pytest.mark.parametrize("feature", ["temperature", "station_id", "flow_cms"])
def test_non_include_feature_request_fails(feature):
    with pytest.raises(FeatureContractError):
        enforce_feature_contract(
            {"NWM_flow", "temperature", "station_id", "flow_cms"}, make_registry(), requested_features=[feature]
        )


def test_unregistered_dataset_column_fails():
    with pytest.raises(FeatureContractError, match="unregistered dataset columns"):
        enforce_feature_contract(
            {"NWM_flow", "temperature", "station_id", "flow_cms", "mystery_feature"}, make_registry()
        )


def test_stale_registry_entry_fails():
    """The key improvement over the first scaffold draft: catches a
    registry entry for a column that no longer exists in the data
    (e.g. after a dataset refresh drops a column), not just the
    reverse direction."""
    with pytest.raises(FeatureContractError, match="registry features absent from dataset"):
        enforce_feature_contract({"NWM_flow", "temperature", "station_id"}, make_registry())


def test_duplicate_registry_feature_fails():
    registry = make_registry()
    registry = pd.concat([registry, registry.iloc[[0]]], ignore_index=True)
    with pytest.raises(FeatureContractError, match="duplicate"):
        enforce_feature_contract({"NWM_flow", "temperature", "station_id", "flow_cms"}, registry)


def test_invalid_registry_status_fails():
    registry = make_registry()
    registry.loc[registry["feature_name"] == "temperature", "status"] = "maybe"
    with pytest.raises(FeatureContractError, match="Invalid provenance registry status"):
        enforce_feature_contract({"NWM_flow", "temperature", "station_id", "flow_cms"}, registry)


def test_duplicate_requested_features_fail():
    with pytest.raises(FeatureContractError, match="duplicate"):
        enforce_feature_contract(
            {"NWM_flow", "temperature", "station_id", "flow_cms"},
            make_registry(),
            requested_features=["NWM_flow", "NWM_flow"],
        )


def test_missing_required_registry_column_fails():
    registry = make_registry().drop(columns=["status"])
    with pytest.raises(FeatureContractError, match="missing required column"):
        enforce_feature_contract({"NWM_flow", "temperature", "station_id", "flow_cms"}, registry)


def test_missing_registry_status_fails():
    registry = make_registry()
    registry.loc[registry["feature_name"] == "temperature", "status"] = None
    with pytest.raises(FeatureContractError, match="status contains missing values"):
        enforce_feature_contract({"NWM_flow", "temperature", "station_id", "flow_cms"}, registry)


def test_missing_registry_feature_name_fails():
    registry = make_registry()
    registry.loc[registry["feature_name"] == "temperature", "feature_name"] = None
    with pytest.raises(FeatureContractError, match="feature_name contains missing values"):
        enforce_feature_contract({"NWM_flow", "temperature", "station_id", "flow_cms"}, registry)
