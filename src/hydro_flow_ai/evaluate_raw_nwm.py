"""
Evaluate raw National Water Model (NWM) streamflow against observed discharge.

This module establishes the uncorrected NWM benchmark that later
HYDRO-FLOW-AI residual-correction models must outperform.

Primary extreme-event evaluation uses station-specific Q95/Q99 thresholds.
Pooled/global Q95/Q99 thresholds are retained only as coarse diagnostic
summaries because pooling across watersheds of different sizes can make
"extreme" flow definitions physically misleading.

Expected input columns
----------------------
Required:
    station_id
    datetime
    flow_cms
    NWM_flow

Usage
-----
Direct file path:

    python src/hydro_flow_ai/evaluate_raw_nwm.py \
        --input-path /path/to/data_model_input.parquet

Data-root style:

    python src/hydro_flow_ai/evaluate_raw_nwm.py \
        --data-root /path/to/collaborator/Hydrologic-AI

Environment variable:

    export NWM_DATA_ROOT=/path/to/collaborator/Hydrologic-AI
    python src/hydro_flow_ai/evaluate_raw_nwm.py

Optional:

    python src/hydro_flow_ai/evaluate_raw_nwm.py \
        --data-root /path/to/Hydrologic-AI \
        --metrics-dir results/metrics \
        --min-rows 20 \
        --fail-on-duplicate-site-date

Outputs
-------
results/metrics/raw_nwm_global_diagnostic.csv
results/metrics/raw_nwm_by_station.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT_ENV_VAR = "NWM_DATA_ROOT"
INPUT_RELATIVE = "data_model_input.parquet"

REQUIRED_COLUMNS = {
    "station_id",
    "datetime",
    "flow_cms",
    "NWM_flow",
}

DEFAULT_MIN_ROWS = 20


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used by the evaluation pipeline."""

    input: Path
    metrics_dir: Path = ROOT / "results" / "metrics"


class EvaluationError(RuntimeError):
    """Raised when baseline evaluation cannot proceed safely."""


# ---------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------


def resolve_input_path(
    explicit: Path | None,
    data_root: Path | None,
) -> Path:
    """Resolve the Parquet input location.

    Priority:
        1. --input-path
        2. --data-root
        3. NWM_DATA_ROOT environment variable, resolved in parse_args()
    """

    if explicit is not None:
        return explicit.expanduser().resolve()

    if data_root is not None:
        return (data_root.expanduser() / INPUT_RELATIVE).resolve()

    raise EvaluationError(
        "Could not locate the input Parquet file.\n"
        "Provide one of:\n"
        "  --input-path PATH\n"
        "  --data-root PATH\n"
        f"  export {DATA_ROOT_ENV_VAR}=PATH"
    )


# ---------------------------------------------------------------------
# Schema and integrity checks
# ---------------------------------------------------------------------


def require_columns(
    df: pd.DataFrame,
    columns: set[str],
    name: str,
) -> None:
    """Raise a clear error when required columns are missing."""

    missing = sorted(columns.difference(df.columns))

    if missing:
        raise EvaluationError(
            f"{name} is missing required column(s): {missing}. "
            f"Columns actually present: {sorted(df.columns.tolist())}"
        )


def prepare_data(
    df: pd.DataFrame,
    *,
    fail_on_duplicate_site_date: bool = False,
) -> pd.DataFrame:
    """Normalize key columns and run dataset-level integrity checks."""

    df = df.copy()

    # -------------------------------------------------------------
    # Datetime normalization
    # -------------------------------------------------------------
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce",
    )

    bad_dates = int(df["datetime"].isna().sum())

    if bad_dates:
        raise EvaluationError(
            f"Input contains {bad_dates} invalid or missing datetime value(s)."
        )

    # -------------------------------------------------------------
    # Station ID normalization
    # -------------------------------------------------------------
    df["station_id"] = df["station_id"].astype("string")

    bad_stations = int(df["station_id"].isna().sum())

    if bad_stations:
        raise EvaluationError(
            f"Input contains {bad_stations} missing station_id value(s)."
        )

    # -------------------------------------------------------------
    # Numeric normalization
    # -------------------------------------------------------------
    for col in ["flow_cms", "NWM_flow"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # -------------------------------------------------------------
    # Duplicate station/date detection
    # -------------------------------------------------------------
    duplicate_mask = df.duplicated(
        ["station_id", "datetime"],
        keep=False,
    )

    n_duplicate_rows = int(duplicate_mask.sum())

    if n_duplicate_rows:
        message = (
            f"Found {n_duplicate_rows} row(s) participating in duplicate "
            "(station_id, datetime) combinations."
        )

        if fail_on_duplicate_site_date:
            raise EvaluationError(message)

        logger.warning(message)

    return df


# ---------------------------------------------------------------------
# Finite-value helpers
# ---------------------------------------------------------------------


def finite_pair_mask(
    obs: pd.Series,
    pred: pd.Series,
) -> np.ndarray:
    """Return mask where both observed and predicted values are finite."""

    obs_arr = pd.to_numeric(
        obs,
        errors="coerce",
    ).to_numpy(dtype=float)

    pred_arr = pd.to_numeric(
        pred,
        errors="coerce",
    ).to_numpy(dtype=float)

    return (
        np.isfinite(obs_arr)
        & np.isfinite(pred_arr)
    )


def finite_observed_subset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return rows with finite observed streamflow.

    Threshold estimation must not include NaN or infinite observed values.
    """

    flow = pd.to_numeric(
        df["flow_cms"],
        errors="coerce",
    ).to_numpy(dtype=float)

    return df.loc[np.isfinite(flow)].copy()


# ---------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------


def metrics(
    obs: pd.Series,
    pred: pd.Series,
    *,
    label: str = "",
    min_rows: int = DEFAULT_MIN_ROWS,
) -> dict[str, float | int]:
    """Compute RMSE, MAE, and bias using finite observation pairs.

    Non-finite values are removed pairwise.

    Metrics are reported as NaN when fewer than `min_rows` valid pairs
    remain.
    """

    obs_arr = pd.to_numeric(
        obs,
        errors="coerce",
    ).to_numpy(dtype=float)

    pred_arr = pd.to_numeric(
        pred,
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(obs_arr)
        & np.isfinite(pred_arr)
    )

    obs_clean = obs_arr[valid]
    pred_clean = pred_arr[valid]

    n_total = len(obs_arr)
    n_valid = int(valid.sum())
    n_removed = n_total - n_valid

    result: dict[str, float | int] = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_removed": n_removed,
        "rmse": np.nan,
        "mae": np.nan,
        "bias": np.nan,
    }

    if n_valid < min_rows:
        if n_total:
            logger.warning(
                "%s: only %d valid finite row(s) out of %d; "
                "minimum required is %d. Reporting NaN metrics.",
                label or "regime",
                n_valid,
                n_total,
                min_rows,
            )

        return result

    errors = pred_clean - obs_clean

    result.update(
        {
            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        obs_clean,
                        pred_clean,
                    )
                )
            ),
            "mae": float(
                mean_absolute_error(
                    obs_clean,
                    pred_clean,
                )
            ),
            "bias": float(errors.mean()),
        }
    )

    return result


# ---------------------------------------------------------------------
# Global pooled diagnostics
# ---------------------------------------------------------------------


def evaluate_global(
    df: pd.DataFrame,
    *,
    min_rows: int,
) -> pd.DataFrame:
    """Evaluate pooled/global metrics.

    IMPORTANT:
        Pooled Q95/Q99 thresholds are diagnostic only.
        They are not the primary scientific extreme-flow definition.
    """

    finite_df = finite_observed_subset(df)

    if finite_df.empty:
        raise EvaluationError(
            "No finite observed flow values are available for "
            "global threshold calculation."
        )

    q95 = float(
        finite_df["flow_cms"].quantile(0.95)
    )

    q99 = float(
        finite_df["flow_cms"].quantile(0.99)
    )

    regimes = {
        "ALL": (
            finite_df,
            np.nan,
        ),
        "GLOBAL_Q95+": (
            finite_df[
                finite_df["flow_cms"] >= q95
            ],
            q95,
        ),
        "GLOBAL_Q99+": (
            finite_df[
                finite_df["flow_cms"] >= q99
            ],
            q99,
        ),
    }

    rows = []

    for regime, (subset, threshold) in regimes.items():

        result = metrics(
            subset["flow_cms"],
            subset["NWM_flow"],
            label=f"global/{regime}",
            min_rows=min_rows,
        )

        rows.append(
            {
                "regime": regime,
                "threshold_scope": "pooled_diagnostic",
                "threshold_cms": threshold,
                **result,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Station-specific evaluation
# ---------------------------------------------------------------------


def evaluate_by_station(
    df: pd.DataFrame,
    *,
    min_rows: int,
) -> pd.DataFrame:
    """Evaluate each gauge using its own Q95/Q99 thresholds."""

    rows = []

    grouped = df.groupby(
        "station_id",
        sort=True,
    )

    for station, group in grouped:

        group = (
            group
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        finite_obs = finite_observed_subset(group)

        if finite_obs.empty:
            logger.warning(
                "Skipping station %s: no finite observed streamflow.",
                station,
            )
            continue

        # ---------------------------------------------------------
        # Station-specific extreme thresholds
        # ---------------------------------------------------------
        station_q95 = float(
            finite_obs["flow_cms"].quantile(0.95)
        )

        station_q99 = float(
            finite_obs["flow_cms"].quantile(0.99)
        )

        q95_subset = finite_obs[
            finite_obs["flow_cms"] >= station_q95
        ]

        q99_subset = finite_obs[
            finite_obs["flow_cms"] >= station_q99
        ]

        # ---------------------------------------------------------
        # Overall metrics
        # ---------------------------------------------------------
        overall = metrics(
            group["flow_cms"],
            group["NWM_flow"],
            label=f"station {station}/ALL",
            min_rows=min_rows,
        )

        # ---------------------------------------------------------
        # Extreme metrics
        # ---------------------------------------------------------
        q95_metrics = metrics(
            q95_subset["flow_cms"],
            q95_subset["NWM_flow"],
            label=f"station {station}/Q95+",
            min_rows=min_rows,
        )

        q99_metrics = metrics(
            q99_subset["flow_cms"],
            q99_subset["NWM_flow"],
            label=f"station {station}/Q99+",
            min_rows=min_rows,
        )

        # ---------------------------------------------------------
        # Descriptive values
        # ---------------------------------------------------------
        nwm_numeric = pd.to_numeric(
            group["NWM_flow"],
            errors="coerce",
        )

        finite_nwm = nwm_numeric[
            np.isfinite(
                nwm_numeric.to_numpy(dtype=float)
            )
        ]

        nwm_max = (
            float(finite_nwm.max())
            if not finite_nwm.empty
            else np.nan
        )

        rows.append(
            {
                "station_id": station,

                "start": group["datetime"].min(),
                "end": group["datetime"].max(),

                "station_rows": len(group),

                "obs_mean_cms": float(
                    finite_obs["flow_cms"].mean()
                ),

                "obs_max_cms": float(
                    finite_obs["flow_cms"].max()
                ),

                "nwm_max_cms": nwm_max,

                # Overall
                "overall_n_total": overall["n_total"],
                "overall_n_valid": overall["n_valid"],
                "overall_n_removed": overall["n_removed"],
                "overall_bias_cms": overall["bias"],
                "overall_mae_cms": overall["mae"],
                "overall_rmse_cms": overall["rmse"],

                # Q95
                "q95_cms": station_q95,
                "q95_n_total": q95_metrics["n_total"],
                "q95_n_valid": q95_metrics["n_valid"],
                "q95_n_removed": q95_metrics["n_removed"],
                "q95_bias_cms": q95_metrics["bias"],
                "q95_mae_cms": q95_metrics["mae"],
                "q95_rmse_cms": q95_metrics["rmse"],
                "q95_sufficient_samples": (
                    q95_metrics["n_valid"] >= min_rows
                ),

                # Q99
                "q99_cms": station_q99,
                "q99_n_total": q99_metrics["n_total"],
                "q99_n_valid": q99_metrics["n_valid"],
                "q99_n_removed": q99_metrics["n_removed"],
                "q99_bias_cms": q99_metrics["bias"],
                "q99_mae_cms": q99_metrics["mae"],
                "q99_rmse_cms": q99_metrics["rmse"],
                "q99_sufficient_samples": (
                    q99_metrics["n_valid"] >= min_rows
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise EvaluationError(
            "No stations could be evaluated."
        )

    return result.sort_values(
        "q95_rmse_cms",
        ascending=False,
        na_position="last",
    )


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def run(
    paths: Paths,
    *,
    min_rows: int,
    fail_on_duplicate_site_date: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete raw-NWM baseline evaluation."""

    if not paths.input.exists():
        raise EvaluationError(
            f"Input file does not exist: {paths.input}"
        )

    logger.info(
        "Loading input dataset: %s",
        paths.input,
    )

    df = pd.read_parquet(paths.input)

    require_columns(
        df,
        REQUIRED_COLUMNS,
        "Input dataframe",
    )

    df = prepare_data(
        df,
        fail_on_duplicate_site_date=(
            fail_on_duplicate_site_date
        ),
    )

    logger.info(
        "Loaded %d rows across %d station(s).",
        len(df),
        df["station_id"].nunique(),
    )

    # -------------------------------------------------------------
    # Missing / non-finite diagnostics
    # -------------------------------------------------------------
    flow_numeric = pd.to_numeric(
        df["flow_cms"],
        errors="coerce",
    ).to_numpy(dtype=float)

    nwm_numeric = pd.to_numeric(
        df["NWM_flow"],
        errors="coerce",
    ).to_numpy(dtype=float)

    n_bad_flow = int(
        (~np.isfinite(flow_numeric)).sum()
    )

    n_bad_nwm = int(
        (~np.isfinite(nwm_numeric)).sum()
    )

    if n_bad_flow or n_bad_nwm:
        logger.warning(
            "Input contains non-finite values: "
            "%d in flow_cms, %d in NWM_flow. "
            "These will be removed pairwise within each metric regime.",
            n_bad_flow,
            n_bad_nwm,
        )

    # -------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------
    global_metrics = evaluate_global(
        df,
        min_rows=min_rows,
    )

    station_metrics = evaluate_by_station(
        df,
        min_rows=min_rows,
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------
    paths.metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_path = (
        paths.metrics_dir
        / "raw_nwm_global_diagnostic.csv"
    )

    station_path = (
        paths.metrics_dir
        / "raw_nwm_by_station.csv"
    )

    global_metrics.to_csv(
        global_path,
        index=False,
    )

    station_metrics.to_csv(
        station_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Sample sufficiency diagnostics
    # -------------------------------------------------------------
    insufficient_q95 = int(
        (
            ~station_metrics[
                "q95_sufficient_samples"
            ]
        ).sum()
    )

    insufficient_q99 = int(
        (
            ~station_metrics[
                "q99_sufficient_samples"
            ]
        ).sum()
    )

    # -------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------
    logger.info(
        "\nGLOBAL RAW NWM "
        "(pooled thresholds — diagnostic only)\n%s",
        global_metrics.to_string(index=False),
    )

    display_columns = [
        "station_id",
        "station_rows",
        "overall_bias_cms",
        "overall_rmse_cms",
        "q95_cms",
        "q95_n_valid",
        "q95_rmse_cms",
        "q99_cms",
        "q99_n_valid",
        "q99_rmse_cms",
    ]

    logger.info(
        "\nWorst 20 stations by station-specific Q95 RMSE\n%s",
        station_metrics[
            display_columns
        ]
        .head(20)
        .to_string(index=False),
    )

    logger.info(
        "Stations below Q95 minimum sample requirement: %d",
        insufficient_q95,
    )

    logger.info(
        "Stations below Q99 minimum sample requirement: %d",
        insufficient_q99,
    )

    logger.info(
        "Saved global diagnostics: %s",
        global_path,
    )

    logger.info(
        "Saved station metrics: %s",
        station_path,
    )

    return (
        global_metrics,
        station_metrics,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:

    env_root = os.environ.get(
        DATA_ROOT_ENV_VAR
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(
            Path(env_root)
            if env_root
            else None
        ),
        help=(
            "Directory containing data_model_input.parquet. "
            f"Defaults to ${DATA_ROOT_ENV_VAR} when defined."
        ),
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help=(
            "Direct path to data_model_input.parquet. "
            "Overrides --data-root."
        ),
    )

    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=ROOT / "results" / "metrics",
        help="Directory for CSV metric outputs.",
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=DEFAULT_MIN_ROWS,
        help=(
            "Minimum number of valid observation/prediction "
            "pairs required to report metrics for a regime."
        ),
    )

    parser.add_argument(
        "--fail-on-duplicate-site-date",
        action="store_true",
        help=(
            "Treat duplicate (station_id, datetime) rows as "
            "a fatal data-integrity error."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    if args.min_rows < 2:
        parser.error(
            "--min-rows must be at least 2."
        )

    return args


def main() -> None:
    """CLI entry point."""

    args = parse_args()

    logging.basicConfig(
        level=(
            logging.DEBUG
            if args.verbose
            else logging.INFO
        ),
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        ),
        datefmt="%H:%M:%S",
    )

    try:

        input_path = resolve_input_path(
            args.input_path,
            args.data_root,
        )

        paths = Paths(
            input=input_path,
            metrics_dir=args.metrics_dir,
        )

        run(
            paths,
            min_rows=args.min_rows,
            fail_on_duplicate_site_date=(
                args.fail_on_duplicate_site_date
            ),
        )

    except EvaluationError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
