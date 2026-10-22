from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Split:
    train: Any
    validation: Any
    test: Any


@dataclass(frozen=True)
class MetricBundle:
    kge: float
    nse: float
    rmse: float
    mae: float
    percent_bias: float
    correlation: float
    variability_ratio: float
    bias_ratio: float

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()

