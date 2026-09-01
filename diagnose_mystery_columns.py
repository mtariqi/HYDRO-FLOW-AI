"""
Self-check before escalating the min/max/mean/median/s1/s2 column mystery.

Two independent checks:
  1. Does the parquet file itself carry embedded schema/column metadata
     that already documents these columns? (cheap, sometimes just works)
  2. Statistical fingerprinting: do these columns behave like a per-station
     historical climatology (constant within a station -- a serious
     leakage red flag if computed from the target itself), or like a
     genuinely time-varying signal? What do they correlate with?

Run from your project root:
    python diagnose_mystery_columns.py
"""
import pandas as pd
import pyarrow.parquet as pq

PATH = "data/raw/collaborator/Hydrologic-AI/data_model_input.parquet"
MYSTERY_COLS = ["min", "max", "mean", "median", "s1", "s2"]

print("=" * 70)
print("CHECK 1: Embedded parquet schema metadata")
print("=" * 70)

pf = pq.ParquetFile(PATH)
schema = pf.schema_arrow

print("\n-- File-level metadata --")
if schema.metadata:
    for k, v in schema.metadata.items():
        print(f"  {k!r} -> {v!r}")
else:
    print("  (none found)")

print("\n-- Column-level metadata for the mystery columns --")
found_any = False
for field in schema:
    if field.name in MYSTERY_COLS:
        if field.metadata:
            print(f"  {field.name}: {field.metadata}")
            found_any = True
        else:
            print(f"  {field.name}: (no metadata)")
if not found_any:
    print("\n  No embedded documentation found for any mystery column.")
    print("  Moving to statistical fingerprinting below.")

print()
print("=" * 70)
print("CHECK 2: Statistical fingerprinting")
print("=" * 70)

df = pd.read_parquet(PATH)

print("\n-- Basic stats for the six mystery columns --")
print(df[MYSTERY_COLS].describe().to_string())

print("\n-- CRITICAL CHECK: are these constant within a station? --")
print("(If yes, this strongly suggests a per-station historical")
print(" climatology -- a serious leakage risk if it was computed")
print(" from the target itself using the full record.)")
for col in MYSTERY_COLS:
    n_unique_per_station = df.groupby("station_id")[col].nunique()
    all_constant = (n_unique_per_station <= 1).all()
    print(f"  {col}: constant within every station? {all_constant} "
          f"(unique values per station: min={n_unique_per_station.min()}, "
          f"max={n_unique_per_station.max()})")

print("\n-- Do min <= median <= mean <= max hold row-wise? --")
print("(Checks whether these four form a coherent summary of ONE thing)")
ordering_ok = ((df["min"] <= df["median"]) & (df["median"] <= df["max"])).mean()
print(f"  min <= median <= max holds for {ordering_ok:.1%} of rows")
mean_between = ((df["min"] <= df["mean"]) & (df["mean"] <= df["max"])).mean()
print(f"  min <= mean <= max holds for {mean_between:.1%} of rows")

print("\n-- Correlation of mystery columns with known columns --")
known_cols = ["flow_cms", "NWM_flow", "precipitation", "temperature", "storage", "swe"]
known_cols = [c for c in known_cols if c in df.columns]
corr = df[MYSTERY_COLS + known_cols].corr().loc[MYSTERY_COLS, known_cols]
print(corr.round(3).to_string())

print("\n-- s1 vs s2: near-identical (two versions of the same thing?) or distinct? --")
print(f"  correlation(s1, s2) = {df['s1'].corr(df['s2']):.4f}")
print(f"  mean absolute difference = {(df['s1'] - df['s2']).abs().mean():.4f}")

print("\n-- Value ranges, for scale comparison against known variables --")
for col in MYSTERY_COLS + known_cols:
    print(f"  {col}: min={df[col].min():.3f}, max={df[col].max():.3f}, mean={df[col].mean():.3f}")
