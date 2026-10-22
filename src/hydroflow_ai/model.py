from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import FEATURES


@dataclass
class ResidualCorrector:
    model: HistGradientBoostingRegressor
    features: list[str]

    @classmethod
    def create(cls, seed: int = 42) -> "ResidualCorrector":
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
        return cls(model=model, features=list(FEATURES))

    def fit(self, frame: pd.DataFrame) -> "ResidualCorrector":
        residual = frame["observed_flow"].to_numpy() - frame["nwm_flow"].to_numpy()
        sample_weight = np.where(
            frame["observed_flow"].to_numpy() >= frame["observed_flow"].quantile(0.95), 3.0, 1.0
        )
        self.model.fit(frame[self.features], residual, sample_weight=sample_weight)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        correction = self.model.predict(frame[self.features])
        return np.maximum(0.0, frame["nwm_flow"].to_numpy() + correction)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "ResidualCorrector":
        return joblib.load(path)

