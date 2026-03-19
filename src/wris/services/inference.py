from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path

from wris.features.preprocessing import row_to_features
from wris.models.training import KNNRegressor


@dataclass
class RiskPrediction:
    burned_area_estimate: float
    risk_score: float
    risk_level: str


class RiskService:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        with self.model_path.open("rb") as f:
            self.model: KNNRegressor = pickle.load(f)

    @staticmethod
    def _risk_level(score: float) -> str:
        if score < 0.33:
            return "Low"
        if score < 0.66:
            return "Moderate"
        return "High"

    def predict(self, features: dict[str, str | float | int]) -> RiskPrediction:
        vector = row_to_features(features)
        burned_area = float(self.model.predict([vector])[0])
        risk_score = max(0.0, min(1.0, 1 - math.exp(-burned_area / 20.0)))
        return RiskPrediction(
            burned_area_estimate=burned_area,
            risk_score=risk_score,
            risk_level=self._risk_level(risk_score),
        )
