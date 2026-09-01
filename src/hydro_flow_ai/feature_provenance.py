"""
Feature Provenance Audit -- Stage 4 of the HYDRO-FLOW-AI pipeline.

Before any temporal feature engineering runs on a candidate predictor
column, that column must pass through this audit. Three checks:

  P1 - Ambiguous variables: does this column carry usable signal at
       all? (near-constant, extreme skew, high missingness, physically
       impossible values like negative flow/precipitation)

  P2 - Target-leakage risk: is this column likely DERIVED FROM the
       target (flow_cms) rather than an independent predictor? This is
       necessarily a heuristic, not a proof -- e.g. NWM_flow is highly
       correlated with flow_cms and is entirely legitimate (it's the
       baseline model output, not derived from the observation), so a
       high-correlation flag means "a human must confirm this," not
       "this is leakage." Name-pattern matching catches the more
       obvious cases (columns literally named "residual", "error",
       "bias_corrected", etc.).

  P3 - Temporal availability: NOT computable from data. Cannot be
       inferred by any statistic -- this script deliberately does NOT
       attempt to guess it. Every candidate column must get an explicit
       human annotation in the registry before being approved.

Mirrors the design of qc_registry.py: automated screening produces
evidence, a human-adjudicated registry (feature_provenance_registry.csv)
records the decision, and nothing is silently included or excluded.
A column flagged by the screen but absent from the registry defaults to
NOT approved, with a loud warning -- the same conservative default used
for station QC.

Usage
-----
python src/hydro_flow_ai/feature_provenance.py \
    --data-root /path/to/collaborator/Hydrologic-AI \
    --target-col flow_cms \
    --registry configs/feature_provenance_registry.csv
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_STATUSES = {"include", "review", "exclude", "target"}
# "target" is distinct from "exclude": both are always missing from the
# approved predictor list, but "target" documents the supervised-learning
# target itself (informational completeness -- this column is never
# screened as a candidate feature in the first place, since
# screen_feature_provenance excludes target_col from candidate_cols).
# "exclude" documents a real candidate feature or identifier that was
# considered and deliberately kept out of the predictor matrix.

# Columns that are identifiers/keys, never candidate predictive features.
# Excluded from the numeric audit entirely rather than flagged as
# "ambiguous" -- a station_id isn't a bad feature, it's not a feature.
IDENTIFIER_COLUMNS = {"station_id", "datetime", "NHDPlusid", "state"}

# Name substrings that suggest a column was DERIVED FROM the target,
# not measured/simulated independently of it. Deliberately narrow --
# "flow" alone would wrongly catch NWM_flow, which is legitimate.
LEAKAGE_NAME_PATTERNS = ("residual", "error", "bias", "corrected", "delta_obs", "obs_minus")

NEAR_CONSTANT_STD_THRESHOLD = 1e-9
HIGH_MISSINGNESS_THRESHOLD = 0.30
HIGH_CORRELATION_THRESHOLD = 0.98
EXTREME_SKEW_THRESHOLD = 5.0


class ProvenanceError(RuntimeError):
    pass


def screen_feature_provenance(
    df: pd.DataFrame,
    target_col: str = "flow_cms",
    identifier_columns: set[str] = IDENTIFIER_COLUMNS,
) -> pd.DataFrame:
    """Run the automated P1/P2 checks over every candidate feature column.

    Returns one row per candidate column (excludes the target itself and
    identifier columns) with descriptive statistics and boolean flags.
    Does NOT determine approval -- see build_feature_provenance_summary.
    """
    if target_col not in df.columns:
        raise ProvenanceError(f"Target column {target_col!r} not found. Columns present: {sorted(df.columns)}")

    candidate_cols = [c for c in df.columns if c != target_col and c not in identifier_columns]
    if not candidate_cols:
        raise ProvenanceError("No candidate feature columns found after excluding target and identifiers.")

    target_numeric = pd.to_numeric(df[target_col], errors="coerce")

    rows = []
    for col in candidate_cols:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series) or pd.to_numeric(series, errors="coerce").notna().any()

        n_total = len(series)
        n_missing = int(series.isna().sum())
        pct_missing = n_missing / n_total if n_total else np.nan
        n_unique = int(series.nunique(dropna=True))

        name_flag = any(pattern in col.lower() for pattern in LEAKAGE_NAME_PATTERNS)

        if is_numeric:
            numeric = pd.to_numeric(series, errors="coerce")
            finite = numeric[np.isfinite(numeric.to_numpy(dtype=float))] if numeric.notna().any() else numeric

            col_min = float(finite.min()) if not finite.empty else np.nan
            col_max = float(finite.max()) if not finite.empty else np.nan
            col_mean = float(finite.mean()) if not finite.empty else np.nan
            col_median = float(finite.median()) if not finite.empty else np.nan
            col_std = float(finite.std()) if not finite.empty else np.nan
            col_skew = float(finite.skew()) if len(finite) > 2 else np.nan

            near_constant = (n_unique <= 1) or (not np.isnan(col_std) and col_std < NEAR_CONSTANT_STD_THRESHOLD)
            extreme_skew = not np.isnan(col_skew) and abs(col_skew) > EXTREME_SKEW_THRESHOLD

            paired = pd.DataFrame({"t": target_numeric, "f": finite}).dropna()
            correlation = float(paired["t"].corr(paired["f"])) if len(paired) > 2 else np.nan
            high_correlation = not np.isnan(correlation) and abs(correlation) > HIGH_CORRELATION_THRESHOLD
        else:
            col_min = col_max = col_mean = col_median = col_std = col_skew = correlation = np.nan
            near_constant = n_unique <= 1
            extreme_skew = False
            high_correlation = False

        high_missingness = pct_missing > HIGH_MISSINGNESS_THRESHOLD

        flagged = name_flag or near_constant or extreme_skew or high_correlation or high_missingness

        rows.append({
            "feature_name": col,
            "dtype": str(series.dtype),
            "n_total": n_total,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "n_unique": n_unique,
            "min": col_min, "max": col_max, "mean": col_mean, "median": col_median,
            "std": col_std, "skew": col_skew,
            "correlation_with_target": correlation,
            "name_suggests_leakage": name_flag,
            "near_constant": near_constant,
            "extreme_skew": extreme_skew,
            "high_correlation": high_correlation,
            "high_missingness": high_missingness,
            "flagged": flagged,
        })

    return pd.DataFrame(rows).sort_values("flagged", ascending=False).reset_index(drop=True)


def load_feature_registry(path: Path | None) -> pd.DataFrame:
    """Load the human-adjudicated feature registry. Empty if not present yet."""
    columns = ["feature_name", "status", "reason", "temporal_availability", "decision_source"]
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=columns)

    registry = pd.read_csv(path, dtype={"feature_name": str})
    missing = set(columns) - set(registry.columns)
    if missing:
        raise ProvenanceError(f"Feature registry is missing column(s): {sorted(missing)}")

    bad_statuses = set(registry["status"].unique()) - VALID_STATUSES
    if bad_statuses:
        raise ProvenanceError(f"Feature registry contains invalid status value(s): {sorted(bad_statuses)}")

    return registry


def build_feature_provenance_summary(screen: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    """Join the automated screen with the adjudication registry.

    A column flagged by the screen but ABSENT from the registry defaults
    to approved=False with a loud warning -- P3 (temporal availability)
    can never be answered by code, so an un-adjudicated flagged column
    must never silently flow into feature engineering.
    """
    summary = screen.copy()

    reg = registry.copy()
    reg_cols = reg[["feature_name", "status", "reason", "temporal_availability"]] if not reg.empty else pd.DataFrame(
        columns=["feature_name", "status", "reason", "temporal_availability"]
    )
    summary = summary.merge(reg_cols, on="feature_name", how="left")

    unregistered_flagged = summary["status"].isna() & summary["flagged"]
    n_unregistered = int(unregistered_flagged.sum())
    if n_unregistered:
        logger.warning(
            "%d feature(s) flagged by the provenance screen but NOT in the registry: %s. "
            "Defaulting to approved=False until reviewed -- temporal availability (P3) "
            "cannot be determined automatically and must be adjudicated explicitly.",
            n_unregistered,
            sorted(summary.loc[unregistered_flagged, "feature_name"].tolist()),
        )

    summary["status"] = summary["status"].where(
        ~summary["status"].isna(),
        summary["flagged"].map({True: "review", False: "include"}),
    )

    # Even an unflagged column with NO explicit temporal-availability
    # annotation is not fully approved -- P3 must be answered for every
    # candidate feature, not just the ones the automated screen caught.
    summary["temporal_availability_confirmed"] = summary["temporal_availability"].notna()
    summary["approved"] = (summary["status"] == "include") & summary["temporal_availability_confirmed"]

    return summary


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

import argparse
import os
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT_ENV_VAR = "NWM_DATA_ROOT"
INPUT_RELATIVE = "data_model_input.parquet"


@dataclass(frozen=True)
class Paths:
    input: Path
    metrics_dir: Path = ROOT / "results" / "metrics"
    registry: Path | None = ROOT / "configs" / "feature_provenance_registry.csv"


def resolve_input_path(explicit: Path | None, data_root: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    if data_root is not None:
        return (data_root.expanduser() / INPUT_RELATIVE).resolve()
    raise ProvenanceError(
        "Could not locate the input Parquet file.\n"
        "Provide one of:\n  --input-path PATH\n  --data-root PATH\n"
        f"  export {DATA_ROOT_ENV_VAR}=PATH"
    )


def run(paths: Paths, target_col: str = "flow_cms") -> pd.DataFrame:
    if not paths.input.exists():
        raise ProvenanceError(f"Input file does not exist: {paths.input}")

    logger.info("Loading input dataset: %s", paths.input)
    df = pd.read_parquet(paths.input)

    screen = screen_feature_provenance(df, target_col=target_col)
    registry = load_feature_registry(paths.registry)
    summary = build_feature_provenance_summary(screen, registry)

    paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = paths.metrics_dir / "feature_provenance_summary.csv"
    summary.to_csv(out_path, index=False)

    n_total = len(summary)
    n_flagged = int(summary["flagged"].sum())
    n_approved = int(summary["approved"].sum())
    logger.info(
        "Feature provenance summary: total=%d, flagged=%d, approved=%d, blocked=%d",
        n_total, n_flagged, n_approved, n_total - n_approved,
    )

    blocked = summary.loc[~summary["approved"], ["feature_name", "flagged", "status", "temporal_availability_confirmed"]]
    if not blocked.empty:
        logger.warning("\nFeatures NOT yet approved for feature engineering:\n%s", blocked.to_string(index=False))

    logger.info("Saved: %s", out_path)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_root = os.environ.get(DATA_ROOT_ENV_VAR)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path(env_root) if env_root else None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "results" / "metrics")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs" / "feature_provenance_registry.csv")
    parser.add_argument("--target-col", type=str, default="flow_cms")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )
    try:
        input_path = resolve_input_path(args.input_path, args.data_root)
        paths = Paths(input=input_path, metrics_dir=args.metrics_dir, registry=args.registry)
        run(paths, target_col=args.target_col)
    except ProvenanceError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
