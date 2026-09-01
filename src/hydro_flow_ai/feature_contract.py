"""
Fail-closed feature contract enforcement -- PREVIEW/SCAFFOLD ONLY.

Not wired into features.py until the feature-provenance registry is frozen.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import pandas as pd


VALID_STATUSES = frozenset({"include", "review", "exclude", "target"})
REQUIRED_REGISTRY_COLUMNS = frozenset({"feature_name", "status"})


class FeatureContractError(RuntimeError):
    """Raised when dataset features violate the provenance contract."""


def enforce_feature_contract(
    available_columns: Iterable[str],
    registry: pd.DataFrame,
    requested_features: Iterable[str] | None = None,
) -> list[str]:
    missing_registry_columns = REQUIRED_REGISTRY_COLUMNS - set(registry.columns)
    if missing_registry_columns:
        raise FeatureContractError(
            f"Invalid provenance registry: missing required column(s): {sorted(missing_registry_columns)}."
        )

    if registry["feature_name"].isna().any():
        raise FeatureContractError("Invalid provenance registry: feature_name contains missing values.")

    if registry["status"].isna().any():
        raise FeatureContractError("Invalid provenance registry: status contains missing values.")

    duplicate_features = (
        registry.loc[registry["feature_name"].duplicated(keep=False), "feature_name"]
        .astype(str).unique().tolist()
    )
    if duplicate_features:
        raise FeatureContractError(
            f"Invalid provenance registry: duplicate feature_name entries found: {sorted(duplicate_features)}."
        )

    invalid_statuses = set(registry["status"]) - VALID_STATUSES
    if invalid_statuses:
        raise FeatureContractError(
            f"Invalid provenance registry status value(s): {sorted(invalid_statuses)}. "
            f"Allowed statuses are {sorted(VALID_STATUSES)}."
        )

    available = set(available_columns)
    registered = set(registry["feature_name"])
    unregistered = available - registered
    stale_registry_entries = registered - available

    if unregistered or stale_registry_entries:
        problems = []
        if unregistered:
            problems.append(f"unregistered dataset columns={sorted(unregistered)}")
        if stale_registry_entries:
            problems.append(f"registry features absent from dataset={sorted(stale_registry_entries)}")
        raise FeatureContractError(
            "Feature-provenance schema drift detected: " + "; ".join(problems)
            + ". The dataset and registry must have exact column coverage before feature engineering proceeds."
        )

    status_by_feature = registry.set_index("feature_name")["status"].to_dict()
    approved = {feature for feature, status in status_by_feature.items() if status == "include"}

    if requested_features is not None:
        requested = list(requested_features)

        # Using Counter per Dr. Alam's suggested improvement (O(n) instead of O(n^2))
        counts = Counter(requested)
        duplicates = sorted([f for f, c in counts.items() if c > 1])
        if duplicates:
            raise FeatureContractError(f"Requested feature list contains duplicate feature(s): {duplicates}.")

        not_approved = [f for f in requested if f not in approved]
        if not_approved:
            details = {f: status_by_feature.get(f, "NOT IN REGISTRY") for f in not_approved}
            raise FeatureContractError(
                f"Requested feature(s) are not approved for modeling: {details}. "
                "Only status=='include' variables may enter feature engineering."
            )
        return requested

    return sorted(approved)
