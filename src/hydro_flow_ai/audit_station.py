from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--station-id", type=str, required=True)
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)

    g = (
        df[df["station_id"].astype(str) == args.station_id]
        .copy()
        .sort_values("datetime")
    )

    if g.empty:
        raise SystemExit(f"No rows found for station {args.station_id}")

    g["error"] = g["NWM_flow"] - g["flow_cms"]
    g["abs_error"] = g["error"].abs()
    g["ratio_nwm_obs"] = g["NWM_flow"] / g["flow_cms"].replace(0, np.nan)

    print("\nSTATION SUMMARY")
    print("station_id:", args.station_id)
    print("rows:", len(g))
    print("date range:", g.datetime.min(), "->", g.datetime.max())
    print("NHDPlusid:", g.NHDPlusid.unique())
    print("state:", g.state.unique())
    print("lat/lon:")
    print(g[["dec_lat_va", "dec_long_va"]].drop_duplicates().to_string(index=False))

    print("\nFLOW SUMMARY")
    print(
        g[["flow_cms", "NWM_flow", "error", "abs_error"]]
        .describe(percentiles=[.5, .9, .95, .99, .999])
        .to_string()
    )

    print("\nTOP ABSOLUTE ERRORS")
    cols = [
        "datetime",
        "flow_cms",
        "NWM_flow",
        "error",
        "ratio_nwm_obs",
        "precipitation",
        "temperature",
        "storage",
        "swe",
    ]

    print(
        g.nlargest(args.top_n, "abs_error")[cols]
        .to_string(index=False)
    )

    print("\nNWM/OBS RATIO SUMMARY")
    print(
        g["ratio_nwm_obs"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .describe(percentiles=[.5, .9, .95, .99])
        .to_string()
    )

    print("\nMONTHLY ERROR")
    g["month"] = g["datetime"].dt.month
    print(
        g.groupby("month")
        .agg(
            n=("flow_cms", "size"),
            obs_mean=("flow_cms", "mean"),
            nwm_mean=("NWM_flow", "mean"),
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
