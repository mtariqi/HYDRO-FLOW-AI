from __future__ import annotations

import numpy as np
import pandas as pd


def adjacency_from_metadata(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    stations = sorted(frame["station_id"].astype(str).unique())
    index = {station: idx for idx, station in enumerate(stations)}
    adjacency = np.zeros((len(stations), len(stations)), dtype=float)
    edges = frame[["station_id", "downstream_station_id"]].drop_duplicates()
    for upstream, downstream in edges.itertuples(index=False):
        if str(downstream) in index:
            adjacency[index[str(upstream)], index[str(downstream)]] = 1.0
    row_sum = adjacency.sum(axis=1, keepdims=True)
    adjacency = np.divide(adjacency, row_sum, out=np.zeros_like(adjacency), where=row_sum > 0)
    return stations, adjacency


def network_summary(frame: pd.DataFrame) -> dict[str, int]:
    stations, adjacency = adjacency_from_metadata(frame)
    return {"stations": len(stations), "directed_edges": int((adjacency > 0).sum())}

