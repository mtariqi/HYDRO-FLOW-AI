"""
Recover datetime values for NWM-ML training rows.

The original training dataframe (NWM_ML_Training_DF.csv) dropped its
`datetime` and `NWIS_site_id` columns, while retaining site coordinates.

This script reconstructs those fields by:

1. Mapping each training row back to its NWIS site using Lat/Long.
2. Rebuilding a dated source table from Climate.csv + observed USGS flow.
3. Restricting candidates to the same (site, day-of-year), then matching
   on temperature, precipitation, and observed flow using an absolute
   floating-point tolerance (not exact/rounded equality -- training and
   source values can differ by tiny floating-point noise even though
   they represent the same underlying observation).
4. Requiring every original training row to match exactly one candidate.
5. Writing diagnostic rows when reconstruction is ambiguous or incomplete.

The script refuses to write the final reconstructed dataset unless the
mapping is strictly one-to-one.

Usage
-----
python src/hydro_flow_ai/reconstruct_dataset.py

Optional overrides:
python src/hydro_flow_ai/reconstruct_dataset.py \
    --train-path Data/Processed/NWM_ML_Training_DF.csv \
    --climate-path Data/Processed/Climate.csv \
    --flow-path Data/Processed/flow.pkl \
    --output-path data/derived/nwm_ml_dated.csv \
    --diagnostics-path data/derived/reconstruction_diagnostics.csv \
    --site-atol 1e-4 \
    --match-atol 1e-5 \
    --verbose
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MATCH_COLUMNS = (
    "temperature_F",
    "precipitation_in",
    "flow_cfs",
)

# Absolute tolerance for numeric-field matching (temperature/precip/flow).
# Empirically determined: 1e-5 is the smallest tolerance that uniquely
# recovers every training row for both gauges, with zero ambiguous
# (multi-candidate) matches even at a looser 1e-4.
MATCH_ATOL = 1e-5

# Absolute tolerance for site-coordinate matching (Lat/Long -> site ID).
# Unrelated to MATCH_ATOL -- kept separate and named distinctly on the
# CLI (--site-atol vs --match-atol) to avoid conflating two different
# tolerances that happen to both be "atol".
SITE_ATOL = 1e-4

SITE_COORDS: dict[str, tuple[float, float]] = {
    "10133800": (40.75966979, -111.5640912),
    "10133600": (40.68803889, -111.5337194),
}


@dataclass(frozen=True)
class Paths:
    """File locations used by the reconstruction pipeline."""

    train: Path = ROOT / "Data/Processed/NWM_ML_Training_DF.csv"
    climate: Path = ROOT / "Data/Processed/Climate.csv"
    flow: Path = ROOT / "Data/Processed/flow.pkl"

    output: Path = ROOT / "data/derived/nwm_ml_dated.csv"
    diagnostics: Path = ROOT / "data/derived/reconstruction_diagnostics.csv"


class ReconstructionError(RuntimeError):
    """Raised when datetime reconstruction is not strictly one-to-one."""


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------


def require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Ensure required columns exist."""
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ReconstructionError(f"{name} is missing required columns: {missing}")


# ---------------------------------------------------------------------
# Site reconstruction
# ---------------------------------------------------------------------


def assign_site_from_coordinates(
    df: pd.DataFrame,
    site_coords: dict[str, tuple[float, float]] = SITE_COORDS,
    atol: float = SITE_ATOL,
) -> pd.DataFrame:
    """
    Assign NWIS site IDs from retained latitude/longitude coordinates.

    The coordinate comparison uses absolute tolerance only. Relative
    tolerance is disabled intentionally.

    Raises
    ------
    ReconstructionError
        If a row matches zero sites or more than one site.
    """
    require_columns(df, ["Lat", "Long"], "Training dataframe")

    result = df.copy()
    match_counts = pd.Series(0, index=result.index, dtype=int)
    result["NWIS_site_id"] = pd.Series(index=result.index, dtype="object")

    for site, (lat, lon) in site_coords.items():
        lat_match = np.isclose(result["Lat"], lat, atol=atol, rtol=0.0)
        lon_match = np.isclose(result["Long"], lon, atol=atol, rtol=0.0)
        mask = lat_match & lon_match

        match_counts = match_counts + mask.astype(int)
        result.loc[mask, "NWIS_site_id"] = str(site)

    unmatched = int((match_counts == 0).sum())
    ambiguous = int((match_counts > 1).sum())

    if unmatched or ambiguous:
        raise ReconstructionError(
            "Site assignment failed: "
            f"{unmatched} row(s) matched no site and "
            f"{ambiguous} row(s) matched multiple sites. "
            "Check SITE_COORDS or --site-atol."
        )

    result["NWIS_site_id"] = result["NWIS_site_id"].astype(str)
    return result


# ---------------------------------------------------------------------
# Flow data
# ---------------------------------------------------------------------


def load_flow_table(flow_path: Path, sites: list[str]) -> pd.DataFrame:
    """
    Convert the per-site flow dictionary into one long dataframe.

    Expected output columns include:

        datetime
        flow_cfs
        NWIS_site_id
    """
    if not flow_path.exists():
        raise ReconstructionError(f"Flow file does not exist: {flow_path}")

    flow_dict = pd.read_pickle(flow_path)

    if not isinstance(flow_dict, dict):
        raise ReconstructionError("Expected flow.pkl to contain a dictionary.")

    missing_sites = [site for site in sites if site not in flow_dict]

    if missing_sites:
        raise ReconstructionError(f"flow.pkl is missing expected site(s): {missing_sites}")

    frames: list[pd.DataFrame] = []

    for site in sites:
        frame = flow_dict[site].copy().reset_index()

        require_columns(frame, ["datetime", "flow_cfs"], f"Flow dataframe for site {site}")

        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
        frame["NWIS_site_id"] = str(site)

        frames.append(frame)

    flow = pd.concat(frames, ignore_index=True)

    duplicate_mask = flow.duplicated(subset=["NWIS_site_id", "datetime"], keep=False)

    if duplicate_mask.any():
        raise ReconstructionError(
            "Observed-flow table contains "
            f"{int(duplicate_mask.sum())} duplicate (site, datetime) rows."
        )

    return flow


# ---------------------------------------------------------------------
# Source table
# ---------------------------------------------------------------------


def build_source_table(climate_path: Path, flow_path: Path, sites: list[str]) -> pd.DataFrame:
    """
    Build the dated source-of-truth table.

    Climate and observed USGS streamflow are joined on:

        datetime
        NWIS_site_id
    """
    if not climate_path.exists():
        raise ReconstructionError(f"Climate file does not exist: {climate_path}")

    climate = pd.read_csv(climate_path).drop(columns=["Unnamed: 0"], errors="ignore")

    require_columns(
        climate,
        ["datetime", "NWIS_site_id", "temperature_F", "precipitation_in"],
        "Climate dataframe",
    )

    climate["datetime"] = pd.to_datetime(climate["datetime"], errors="raise")
    climate["NWIS_site_id"] = climate["NWIS_site_id"].astype(str)
    climate = climate[climate["NWIS_site_id"].isin(sites)].copy()

    climate_duplicates = climate.duplicated(subset=["NWIS_site_id", "datetime"], keep=False)

    if climate_duplicates.any():
        raise ReconstructionError(
            "Climate data contains "
            f"{int(climate_duplicates.sum())} duplicate (site, datetime) rows."
        )

    flow = load_flow_table(flow_path, sites=sites)

    source = climate.merge(
        flow[["datetime", "NWIS_site_id", "flow_cfs"]],
        on=["datetime", "NWIS_site_id"],
        how="inner",
        validate="one_to_one",
    )

    source["DOY"] = source["datetime"].dt.dayofyear

    logger.info("Built dated source table with %d rows.", len(source))
    logger.info(
        "Source rows by site:\n%s",
        source["NWIS_site_id"].value_counts().sort_index(),
    )

    return source


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------


def reconstruct_dates(
    train: pd.DataFrame,
    source: pd.DataFrame,
    atol: float = MATCH_ATOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recover exactly one datetime for every training row.

    Candidate dates are first restricted by (site, day-of-year), which
    keeps the search space small (at most ~1 candidate per year of
    overlapping data). Numeric fields are then compared using an
    absolute tolerance -- not exact or rounded equality -- because
    training and source values can carry tiny floating-point noise
    even when they represent the same underlying observation.

    Returns
    -------
    matched
        One row per original training row that matched exactly once,
        with `datetime` populated.
    diagnostics
        One row per original training row that matched zero times or
        more than once, with a `status` column explaining why.
    """
    require_columns(train, ["_row_id", "NWIS_site_id", "DOY", *MATCH_COLUMNS], "Training dataframe")
    require_columns(source, ["NWIS_site_id", "datetime", "DOY", *MATCH_COLUMNS], "Source dataframe")

    source_groups = {
        key: group for key, group in source.groupby(["NWIS_site_id", "DOY"], sort=False)
    }

    matched_rows = []
    diagnostic_rows = []

    for _, row in train.iterrows():
        key = (row["NWIS_site_id"], row["DOY"])
        candidates = source_groups.get(key)

        if candidates is None:
            diagnostic_rows.append(
                {
                    "_row_id": row["_row_id"],
                    "NWIS_site_id": row["NWIS_site_id"],
                    "DOY": row["DOY"],
                    "status": "no_site_doy_candidate",
                    "n_candidates": 0,
                }
            )
            continue

        mask = np.ones(len(candidates), dtype=bool)

        for col in MATCH_COLUMNS:
            mask &= np.isclose(
                candidates[col].to_numpy(dtype=float),
                float(row[col]),
                atol=atol,
                rtol=0.0,
                equal_nan=False,
            )

        hits = candidates.loc[mask]

        if len(hits) == 1:
            result = row.to_dict()
            result["datetime"] = hits.iloc[0]["datetime"]
            matched_rows.append(result)
        else:
            diagnostic_rows.append(
                {
                    "_row_id": row["_row_id"],
                    "NWIS_site_id": row["NWIS_site_id"],
                    "DOY": row["DOY"],
                    "status": "zero_match" if len(hits) == 0 else "multiple_matches",
                    "n_candidates": len(hits),
                }
            )

    matched = pd.DataFrame(matched_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)

    logger.info("Original training rows : %d", len(train))
    logger.info("Unique matched rows    : %d", len(matched))
    logger.info("Problem row IDs        : %d", len(diagnostics))

    if not diagnostics.empty:
        logger.info(
            "Problem breakdown:\n%s",
            diagnostics["status"].value_counts(),
        )

    return matched, diagnostics


# ---------------------------------------------------------------------
# Strict validation
# ---------------------------------------------------------------------


def validate_one_to_one(
    train: pd.DataFrame,
    matched: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    """Require exactly one recovered datetime per original training row."""
    problems: list[str] = []

    if not diagnostics.empty:
        problems.append(f"{len(diagnostics)} row(s) failed to match uniquely")

    if len(matched) != len(train):
        problems.append(f"{len(matched)} matched rows found; expected {len(train)}")

    if not matched.empty:
        row_id_counts = matched["_row_id"].value_counts()
        bad_row_ids = row_id_counts[row_id_counts != 1]

        if not bad_row_ids.empty:
            problems.append(
                f"{len(bad_row_ids)} original row ID(s) appear more than once in matched output"
            )

        duplicate_mask = matched.duplicated(subset=["NWIS_site_id", "datetime"], keep=False)

        if duplicate_mask.any():
            problems.append(
                f"{int(duplicate_mask.sum())} rows share duplicate (site, datetime) values"
            )

    if problems:
        raise ReconstructionError("Reconstruction is not one-to-one: " + "; ".join(problems))


# ---------------------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------------------


def clean_reconstructed_table(matched: pd.DataFrame) -> pd.DataFrame:
    """Remove temporary reconstruction columns and sort chronologically."""
    result = matched.drop(columns=["_row_id"], errors="ignore").copy()
    result["datetime"] = pd.to_datetime(result["datetime"])
    result = result.sort_values(["NWIS_site_id", "datetime"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def run(
    paths: Paths,
    site_atol: float = SITE_ATOL,
    match_atol: float = MATCH_ATOL,
) -> pd.DataFrame:
    """Run full datetime reconstruction pipeline."""
    if not paths.train.exists():
        raise ReconstructionError(f"Training file does not exist: {paths.train}")

    train = pd.read_csv(paths.train).drop(columns=["Unnamed: 0"], errors="ignore")
    train["_row_id"] = np.arange(len(train))
    train = assign_site_from_coordinates(train, atol=site_atol)

    sites = sorted(train["NWIS_site_id"].unique().tolist())

    logger.info("Training sites: %s", sites)
    logger.info(
        "Original rows by site:\n%s",
        train["NWIS_site_id"].value_counts().sort_index(),
    )

    source = build_source_table(paths.climate, paths.flow, sites=sites)

    matched, diagnostics = reconstruct_dates(train, source, atol=match_atol)

    paths.output.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale diagnostics from an earlier failed run.
    if diagnostics.empty:
        paths.diagnostics.unlink(missing_ok=True)
    else:
        diagnostics.to_csv(paths.diagnostics, index=False)
        logger.warning("Wrote reconstruction diagnostics to %s", paths.diagnostics)

    validate_one_to_one(train, matched, diagnostics)

    matched = clean_reconstructed_table(matched)
    matched.to_csv(paths.output, index=False)

    logger.info("SUCCESS: exact one-to-one reconstruction.")
    logger.info("Saved reconstructed dataset to %s", paths.output)

    summary = matched.groupby("NWIS_site_id")["datetime"].agg(["min", "max", "count"])

    logger.info("Recovered date ranges:\n%s", summary)
    logger.info("Total reconstructed rows: %d", len(matched))

    return matched


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = Paths()

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train-path", type=Path, default=defaults.train)
    parser.add_argument("--climate-path", type=Path, default=defaults.climate)
    parser.add_argument("--flow-path", type=Path, default=defaults.flow)
    parser.add_argument("--output-path", type=Path, default=defaults.output)
    parser.add_argument("--diagnostics-path", type=Path, default=defaults.diagnostics)
    parser.add_argument(
        "--site-atol",
        type=float,
        default=SITE_ATOL,
        help="Absolute tolerance for matching Lat/Long to a known site.",
    )
    parser.add_argument(
        "--match-atol",
        type=float,
        default=MATCH_ATOL,
        help=(
            "Absolute tolerance for matching temperature/precipitation/flow "
            "between training rows and the reconstructed source table."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = Paths(
        train=args.train_path,
        climate=args.climate_path,
        flow=args.flow_path,
        output=args.output_path,
        diagnostics=args.diagnostics_path,
    )

    try:
        run(paths, site_atol=args.site_atol, match_atol=args.match_atol)
    except ReconstructionError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
