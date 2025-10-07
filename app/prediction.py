from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from . import features
from .config import MODEL_PATH


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


@dataclass
class ModelArtifact:
    feature_names: List[str]
    weights: List[float]
    numeric_stats: Dict[str, List[float]]


class ChurnModel:
    """Wrapper class for the churn prediction pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._artifact: ModelArtifact | None = None

    def load_artifact(self) -> ModelArtifact:
        if self._artifact is None:
            with self.model_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            numeric_stats = {k: list(v) for k, v in payload["numeric_stats"].items()}
            self._artifact = ModelArtifact(
                feature_names=payload["feature_names"],
                weights=payload["weights"],
                numeric_stats=numeric_stats,
            )
        return self._artifact

    def _vectorize(self, payload: Dict[str, Any]) -> List[float]:
        artifact = self.load_artifact()
        feature_vector = [1.0]
        for column in features.FEATURES.numerical:
            mean, std = artifact.numeric_stats[column]
            value = payload.get(column, 0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            feature_vector.append((value - mean) / std)

        for column in features.FEATURES.categorical:
            values = [name.split("__", 1)[1] for name in artifact.feature_names if name.startswith(f"{column}__")]
            current_value = str(payload.get(column, ""))
            for value in values:
                feature_vector.append(1.0 if current_value == value else 0.0)
        return feature_vector

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        artifact = self.load_artifact()
        vector = self._vectorize(payload)
        score = sum(w * x for w, x in zip(artifact.weights, vector))
        probability = sigmoid(score)
        prediction = "Yes" if probability >= 0.5 else "No"
        return {"prediction": prediction, "probability": probability}


churn_model = ChurnModel()
