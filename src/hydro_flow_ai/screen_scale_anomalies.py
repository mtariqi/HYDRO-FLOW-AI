"""Screen stations for severe NWM-versus-observed scale anomalies.

This module identifies stations where raw National Water Model (NWM)
streamflow and observed discharge occupy implausibly different scales.

The screen is intended as a data-quality / hydrologic-diagnostics step
before pooled machine-learning training. A flagged station is NOT
automatically considered erroneous. Potential explanations include:

- incorrect NWM reach-to-gauge crosswalk;
- strong regulation, diversion, or reservoir operations;
- incompatible flow definitions or units;
- severe structural NWM bias;
- preprocessing or data-integration problems.

Flags should trigger investigation rather than automatic exclusion.

Expected input
--------------
A multi-station Parquet dataset containing at least:

    station_id
    flow_cms
    NWM_flow

Usage
-----
Data root:

    python src/hydro_flow_ai/screen_scale_anomalies.py \
        --data-root /path/to/Hydrologic-AI

Environment variable:

    export NWM_DATA_ROOT=/path/to/Hydrologic-AI
    python src/hydro_flow_ai/screen_scale_anomalies.py

Direct file path:

    python src/hydro_flow_ai/screen_scale_anomalies.py \
        --input-path /path/to/data_model_input.parquet

Optional:

    python src/hydro_flow_ai/screen_scale_anomalies.py \
        --data-root /path/to/Hydrologic-AI \
        --ratio-threshold 5 \
        --min-rows 20 \
        --lower-quantile 0.05 \
        --upper-quantile 0.95 \
        -v
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT_ENV_VAR = "NWM_DATA_ROOT"
INPUT_RELATIVE = "data_model_input.parquet"

DEFAULT_OUTPUT_PATH = (
    ROOT
    / "results"
    / "metrics"
    / "nwm_scale_anomaly_screen.csv"
)

REQUIRED_COLUMNS = {
    "station_id",
    "flow_cms",
    "NWM_flow",
}


class ScaleAuditError(RuntimeError):
    """Raised when the scale-anomaly audit cannot run safely."""


def resolve_input_path(
    explicit: Path | None,
    data_root: Path | None,
) -> Path:
    """Resolve input Parquet path using the project CLI convention.

    Priority:
        1. --input-path
        2. --data-root
        3. NWM_DATA_ROOT environment variable, resolved in parse_args()
    """

    if explicit is not None:
        return explicit.expanduser().resolve()

    if data_root is not None:
        return (
            data_root.expanduser()
            / INPUT_RELATIVE
        ).resolve()

    raise ScaleAuditError(
        "Could not locate the input Parquet file.\n"
        "Provide one of:\n"
        "  --input-path PATH\n"
        "  --data-root PATH\n"
        f"  export {DATA_ROOT_ENV_VAR}=PATH"
    )


def validate_columns(df: pd.DataFrame) -> None:
    """Verify that required input columns exist."""

    missing = sorted(
        REQUIRED_COLUMNS.difference(df.columns)
    )

    if missing:
        raise ScaleAuditError(
            "Input dataset is missing required column(s): "
            + ", ".join(missing)
        )


def finite_values(
    series: pd.Series,
) -> np.ndarray:
    """Convert a Series to finite floating-point values."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=float)

    return values[
        np.isfinite(values)
    ]


def safe_ratio(
    numerator: float,
    denominator: float,
    epsilon: float = 1e-12,
) -> float:
    """Return numerator / denominator safely."""

    if abs(denominator) <= epsilon:
        return np.inf

    return numerator / denominator


def intervals_overlap(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
) -> bool:
    """Return True when two closed numeric intervals overlap."""

    return (
        max(low_a, low_b)
        <= min(high_a, high_b)
    )


def screen_for_scale_anomalies(
    df: pd.DataFrame,
    *,
    ratio_threshold: float = 5.0,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_rows: int = 20,
) -> pd.DataFrame:
    """Flag stations with severe NWM-versus-observed scale mismatch.

    Three complementary diagnostics are used.

    1. no_range_overlap
       The complete finite observed and NWM ranges do not intersect.

    2. no_central_range_overlap
       The central quantile intervals (default Q05-Q95) do not intersect.

    3. large_scale_ratio
       Either the mean or median NWM/observed ratio exceeds
       ``ratio_threshold`` in either direction.

    ``central_overlap_fraction`` is defined as:

        overlap_width / enclosing_span_width

    where ``enclosing_span_width`` is the width of the smallest interval
    spanning both central ranges. This is intentionally not called the
    union width, because disjoint intervals include a gap that is not part
    of the mathematical union.

    A station is flagged when any of the three diagnostics is true.

    A flag indicates a station requiring investigation. It does not prove
    that the station data, NWM data, or crosswalk are incorrect.
    """

    validate_columns(df)

    if ratio_threshold <= 1:
        raise ValueError(
            "ratio_threshold must be greater than 1."
        )

    if not (
        0 <= lower_quantile
        < upper_quantile
        <= 1
    ):
        raise ValueError(
            "Quantiles must satisfy "
            "0 <= lower_quantile < upper_quantile <= 1."
        )

    if min_rows < 2:
        raise ValueError(
            "min_rows must be at least 2."
        )

    rows: list[dict[str, object]] = []

    for station, group in df.groupby(
        "station_id",
        sort=True,
    ):
        obs = finite_values(
            group["flow_cms"]
        )

        nwm = finite_values(
            group["NWM_flow"]
        )

        if (
            len(obs) < min_rows
            or len(nwm) < min_rows
        ):
            logger.warning(
                "Station %s has insufficient finite data "
                "(obs=%d, NWM=%d); skipping.",
                station,
                len(obs),
                len(nwm),
            )
            continue

        # ---------------------------------------------------------
        # Descriptive statistics
        # ---------------------------------------------------------

        obs_min = float(np.min(obs))
        obs_max = float(np.max(obs))
        obs_mean = float(np.mean(obs))
        obs_median = float(np.median(obs))

        nwm_min = float(np.min(nwm))
        nwm_max = float(np.max(nwm))
        nwm_mean = float(np.mean(nwm))
        nwm_median = float(np.median(nwm))

        obs_q_low = float(
            np.quantile(
                obs,
                lower_quantile,
            )
        )

        obs_q_high = float(
            np.quantile(
                obs,
                upper_quantile,
            )
        )

        nwm_q_low = float(
            np.quantile(
                nwm,
                lower_quantile,
            )
        )

        nwm_q_high = float(
            np.quantile(
                nwm,
                upper_quantile,
            )
        )

        # ---------------------------------------------------------
        # Full-range and central-range overlap
        # ---------------------------------------------------------

        full_overlap = intervals_overlap(
            obs_min,
            obs_max,
            nwm_min,
            nwm_max,
        )

        central_overlap = intervals_overlap(
            obs_q_low,
            obs_q_high,
            nwm_q_low,
            nwm_q_high,
        )

        no_range_overlap = (
            not full_overlap
        )

        no_central_range_overlap = (
            not central_overlap
        )

        # ---------------------------------------------------------
        # Mean / median scale ratios
        # ---------------------------------------------------------

        mean_ratio = safe_ratio(
            nwm_mean,
            obs_mean,
        )

        median_ratio = safe_ratio(
            nwm_median,
            obs_median,
        )

        reciprocal_threshold = (
            1.0 / ratio_threshold
        )

        large_mean_ratio = (
            mean_ratio > ratio_threshold
            or mean_ratio
            < reciprocal_threshold
        )

        large_median_ratio = (
            median_ratio > ratio_threshold
            or median_ratio
            < reciprocal_threshold
        )

        large_scale_ratio = (
            large_mean_ratio
            or large_median_ratio
        )

        # ---------------------------------------------------------
        # Log-ratio severity
        #
        # log(5) and log(1/5) have equal magnitude,
        # giving a symmetric severity measure.
        # ---------------------------------------------------------

        log_mean_ratio = (
            float(np.log(mean_ratio))
            if (
                np.isfinite(mean_ratio)
                and mean_ratio > 0
            )
            else np.nan
        )

        log_median_ratio = (
            float(np.log(median_ratio))
            if (
                np.isfinite(median_ratio)
                and median_ratio > 0
            )
            else np.nan
        )

        # ---------------------------------------------------------
        # Central-range overlap fraction
        # ---------------------------------------------------------

        overlap_low = max(
            obs_q_low,
            nwm_q_low,
        )

        overlap_high = min(
            obs_q_high,
            nwm_q_high,
        )

        overlap_width = max(
            0.0,
            overlap_high - overlap_low,
        )

        enclosing_span_low = min(
            obs_q_low,
            nwm_q_low,
        )

        enclosing_span_high = max(
            obs_q_high,
            nwm_q_high,
        )

        enclosing_span_width = (
            enclosing_span_high
            - enclosing_span_low
        )

        central_overlap_fraction = (
            overlap_width
            / enclosing_span_width
            if enclosing_span_width > 0
            else 1.0
        )

        # ---------------------------------------------------------
        # Final anomaly flag
        # ---------------------------------------------------------

        flagged = (
            no_range_overlap
            or no_central_range_overlap
            or large_scale_ratio
        )

        rows.append(
            {
                "station_id": station,

                "n_obs": len(obs),
                "n_nwm": len(nwm),

                # Observed distribution
                "obs_min_cms": obs_min,
                "obs_q_low_cms": obs_q_low,
                "obs_median_cms": obs_median,
                "obs_mean_cms": obs_mean,
                "obs_q_high_cms": obs_q_high,
                "obs_max_cms": obs_max,

                # NWM distribution
                "nwm_min_cms": nwm_min,
                "nwm_q_low_cms": nwm_q_low,
                "nwm_median_cms": nwm_median,
                "nwm_mean_cms": nwm_mean,
                "nwm_q_high_cms": nwm_q_high,
                "nwm_max_cms": nwm_max,

                # Scale diagnostics
                "mean_scale_ratio": mean_ratio,
                "median_scale_ratio": median_ratio,

                "log_mean_scale_ratio": (
                    log_mean_ratio
                ),

                "log_median_scale_ratio": (
                    log_median_ratio
                ),

                # Range diagnostics
                "central_overlap_fraction": (
                    central_overlap_fraction
                ),

                "no_range_overlap": (
                    no_range_overlap
                ),

                "no_central_range_overlap": (
                    no_central_range_overlap
                ),

                # Ratio diagnostics
                "large_mean_ratio": (
                    large_mean_ratio
                ),

                "large_median_ratio": (
                    large_median_ratio
                ),

                "large_scale_ratio": (
                    large_scale_ratio
                ),

                "flagged": flagged,
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        raise ScaleAuditError(
            "No stations contained enough finite data "
            "to evaluate."
        )

    # -------------------------------------------------------------
    # Symmetric severity score
    # -------------------------------------------------------------

    result["_severity"] = np.maximum(
        np.abs(
            result[
                "log_mean_scale_ratio"
            ].fillna(0.0)
        ),
        np.abs(
            result[
                "log_median_scale_ratio"
            ].fillna(0.0)
        ),
    )

    result = (
        result
        .sort_values(
            [
                "flagged",
                "_severity",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .drop(
            columns="_severity"
        )
        .reset_index(drop=True)
    )

    return result


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    env_root = os.environ.get(
        DATA_ROOT_ENV_VAR
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
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
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "CSV path for anomaly-screen results."
        ),
    )

    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=5.0,
        help=(
            "Flag mean or median NWM/observed ratios "
            "above this value or below its reciprocal."
        ),
    )

    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.05,
        help=(
            "Lower bound for the central-range "
            "overlap diagnostic."
        ),
    )

    parser.add_argument(
        "--upper-quantile",
        type=float,
        default=0.95,
        help=(
            "Upper bound for the central-range "
            "overlap diagnostic."
        ),
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=20,
        help=(
            "Minimum finite observations required "
            "for a station to be screened."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    if args.ratio_threshold <= 1:
        parser.error(
            "--ratio-threshold must be > 1."
        )

    if not (
        0
        <= args.lower_quantile
        < args.upper_quantile
        <= 1
    ):
        parser.error(
            "--lower-quantile and --upper-quantile must satisfy "
            "0 <= lower < upper <= 1."
        )

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

        if not input_path.exists():
            raise ScaleAuditError(
                f"Input file does not exist: {input_path}"
            )

        logger.info(
            "Loading input dataset: %s",
            input_path,
        )

        df = pd.read_parquet(
            input_path
        )

        logger.info(
            "Loaded %d rows across %d station(s).",
            len(df),
            (
                df["station_id"].nunique()
                if "station_id" in df.columns
                else 0
            ),
        )

        result = screen_for_scale_anomalies(
            df,
            ratio_threshold=(
                args.ratio_threshold
            ),
            lower_quantile=(
                args.lower_quantile
            ),
            upper_quantile=(
                args.upper_quantile
            ),
            min_rows=args.min_rows,
        )

        args.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        result.to_csv(
            args.output_path,
            index=False,
        )

        flagged = result[
            result["flagged"]
        ]

        print()
        print(
            "NWM SCALE-ANOMALY SCREEN"
        )
        print("=" * 72)

        print(
            f"Stations evaluated : {len(result)}"
        )

        print(
            f"Stations flagged   : {len(flagged)}"
        )

        display_columns = [
            "station_id",
            "obs_mean_cms",
            "nwm_mean_cms",
            "mean_scale_ratio",
            "median_scale_ratio",
            "central_overlap_fraction",
            "no_range_overlap",
            "no_central_range_overlap",
            "large_scale_ratio",
            "flagged",
        ]

        if not flagged.empty:

            print(
                "\nFLAGGED STATIONS"
            )

            print(
                flagged[
                    display_columns
                ].to_string(
                    index=False
                )
            )

        else:
            print(
                "\nNo stations were flagged."
            )

        print(
            f"\nSaved: {args.output_path}"
        )

    except ScaleAuditError as exc:
        logger.error(
            "%s",
            exc,
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
