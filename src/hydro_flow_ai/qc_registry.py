"""Exclusion registry and QC-summary provenance table.

Deliberately separates two different decisions that must never be
conflated:

  - DETECTION: screen_scale_anomalies() flags a station algorithmically.
  - ADJUDICATION: the exclusion registry records the human/scientific
    decision about what to do with a flagged station.

A flagged station is never automatically excluded. But a flagged
station that has NOT yet been adjudicated (no registry row at all) must
never silently default to "safe to train on" either -- that would
defeat the entire point of screening for it. So the default logic is:

  - Not in registry, not flagged by the screen  -> include (assumed fine)
  - Not in registry, but flagged by the screen   -> WARN, exclude from
    model_eligible until someone actually adjudicates it
  - In registry with status="include"            -> include
  - In registry with status="review"/"exclude"    -> exclude from
    model_eligible (both block training; "review" vs "exclude" is a
    human-facing distinction about whether adjudication is still open)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

VALID_STATUSES = {"include", "review", "exclude"}


class RegistryError(RuntimeError):
    pass


def load_exclusion_registry(path: Path | None) -> pd.DataFrame:
    """Load the station exclusion registry CSV.

    Returns an empty (but correctly-columned) DataFrame if no path is
    given or the file doesn't exist yet -- a registry is optional until
    the first station gets flagged.
    """
    columns = ["station_id", "status", "reason", "evidence", "decision_source"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=columns)

    registry = pd.read_csv(path, dtype={"station_id": str})

    missing = set(columns) - set(registry.columns)
    if missing:
        raise RegistryError(f"Exclusion registry is missing column(s): {sorted(missing)}")

    bad_statuses = set(registry["status"].unique()) - VALID_STATUSES
    if bad_statuses:
        raise RegistryError(
            f"Exclusion registry contains invalid status value(s): {sorted(bad_statuses)}. "
            f"Valid values are: {sorted(VALID_STATUSES)}"
        )

    return registry


def build_station_qc_summary(
    station_metrics: pd.DataFrame,
    scale_screen: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    """Join station-level baseline metrics, scale-anomaly evidence, and
    the exclusion registry into one provenance table.

    Every row is one station. `model_eligible` is the single
    authoritative answer to "can this station's data be used in pooled
    training" -- everything else in this table is the evidence trail
    for why.
    """
    station_metrics = station_metrics.copy()
    station_metrics["station_id"] = station_metrics["station_id"].astype(str)

    base = station_metrics[
        ["station_id", "station_rows", "overall_rmse_cms", "overall_bias_cms",
         "q95_rmse_cms", "q99_rmse_cms"]
    ].copy()

    screen = scale_screen.copy()
    screen["station_id"] = screen["station_id"].astype(str)
    screen_cols = screen[
        ["station_id", "mean_scale_ratio", "median_scale_ratio",
         "central_overlap_fraction", "flagged"]
    ].rename(columns={"flagged": "scale_flagged"})

    summary = base.merge(screen_cols, on="station_id", how="left")

    reg = registry.copy()
    if not reg.empty:
        reg["station_id"] = reg["station_id"].astype(str)
    reg_cols = reg[["station_id", "status", "reason"]].rename(
        columns={"status": "registry_status", "reason": "registry_reason"}
    ) if not reg.empty else pd.DataFrame(columns=["station_id", "registry_status", "registry_reason"])

    summary = summary.merge(reg_cols, on="station_id", how="left")

    # Stations never seen by the registry get NaN here -- resolve defaults below.
    unregistered_but_flagged = summary["registry_status"].isna() & summary["scale_flagged"].fillna(False)
    n_unregistered_flagged = int(unregistered_but_flagged.sum())
    if n_unregistered_flagged:
        logger.warning(
            "%d station(s) are flagged by the scale screen but have NO exclusion-registry "
            "entry yet: %s. Defaulting to model_eligible=False until adjudicated -- "
            "add a row to the registry to resolve this.",
            n_unregistered_flagged,
            sorted(summary.loc[unregistered_but_flagged, "station_id"].tolist()),
        )

    # Default status: unregistered + not flagged -> include (assumed fine).
    # unregistered + flagged -> review (forces model_eligible=False below).
    summary["registry_status"] = summary["registry_status"].where(
        ~summary["registry_status"].isna(),
        summary["scale_flagged"].fillna(False).map({True: "review", False: "include"}),
    )

    summary["model_eligible"] = summary["registry_status"] == "include"

    return summary
