from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import features
from .config import DATA_PATH, MODEL_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE


@dataclass
class Dataset:
    feature_names: List[str]
    rows: List[List[float]]
    targets: List[int]
    numeric_stats: Dict[str, Tuple[float, float]]


def load_dataset(path: Path = DATA_PATH) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        header = None
        data: List[Dict[str, str]] = []
        for line in f:
            if header is None:
                header = line.strip().split(",")
                continue
            values = line.strip().split(",")
            if len(values) != len(header):
                continue
            data.append(dict(zip(header, values)))
    return data


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def prepare_dataset(rows: List[Dict[str, str]]) -> Dataset:
    random.seed(RANDOM_STATE)

    numeric_stats: Dict[str, Tuple[float, float]] = {}
    for column in features.FEATURES.numerical:
        values = [_to_float(row.get(column, "0")) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) or 1.0
        numeric_stats[column] = (mean, std)

    category_values: Dict[str, List[str]] = {}
    for column in features.FEATURES.categorical:
        unique_values = sorted({row.get(column, "") for row in rows})
        category_values[column] = unique_values

    feature_names: List[str] = ["bias"]
    feature_names.extend(features.FEATURES.numerical)
    for column, values in category_values.items():
        feature_names.extend([f"{column}__{value}" for value in values])

    processed_rows: List[List[float]] = []
    targets: List[int] = []
    for row in rows:
        vector: List[float] = [1.0]
        for column in features.FEATURES.numerical:
            mean, std = numeric_stats[column]
            value = ( _to_float(row.get(column, "0")) - mean ) / std
            vector.append(value)

        for column in features.FEATURES.categorical:
            values = category_values[column]
            current_value = row.get(column, "")
            for value in values:
                vector.append(1.0 if value == current_value else 0.0)

        processed_rows.append(vector)
        targets.append(1 if row.get(features.TARGET_COLUMN, "No") == "Yes" else 0)

    return Dataset(feature_names, processed_rows, targets, numeric_stats)


def train_test_split(dataset: Dataset) -> Tuple[Dataset, Dataset]:
    indices = list(range(len(dataset.rows)))
    random.shuffle(indices)
    split_index = int(len(indices) * (1 - TEST_SIZE))
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    def subset(idxs: Sequence[int]) -> Dataset:
        return Dataset(
            dataset.feature_names,
            [dataset.rows[i] for i in idxs],
            [dataset.targets[i] for i in idxs],
            dataset.numeric_stats,
        )

    return subset(train_idx), subset(test_idx)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def train_logistic_regression(dataset: Dataset, epochs: int = 300, learning_rate: float = 0.1) -> List[float]:
    weights = [0.0 for _ in dataset.feature_names]
    for _ in range(epochs):
        gradients = [0.0 for _ in dataset.feature_names]
        for features_row, target in zip(dataset.rows, dataset.targets):
            z = sum(w * x for w, x in zip(weights, features_row))
            prediction = sigmoid(z)
            error = prediction - target
            for i in range(len(weights)):
                gradients[i] += error * features_row[i]
        for i in range(len(weights)):
            weights[i] -= learning_rate * gradients[i] / len(dataset.rows)
    return weights


def predict(weights: Sequence[float], row: Sequence[float]) -> float:
    z = sum(w * x for w, x in zip(weights, row))
    return sigmoid(z)


def evaluate(weights: Sequence[float], dataset: Dataset) -> Dict[str, float]:
    probabilities = [predict(weights, row) for row in dataset.rows]
    predictions = [1 if p >= 0.5 else 0 for p in probabilities]
    correct = sum(1 for y_true, y_pred in zip(dataset.targets, predictions) if y_true == y_pred)
    accuracy = correct / len(dataset.targets) if dataset.targets else 0
    return {"accuracy": accuracy}


def train_and_save_model() -> Dict[str, Dict[str, float]]:
    raw_rows = load_dataset()
    dataset = prepare_dataset(raw_rows)
    train_set, test_set = train_test_split(dataset)

    weights = train_logistic_regression(train_set)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_payload = {
        "feature_names": dataset.feature_names,
        "weights": weights,
        "numeric_stats": {k: list(v) for k, v in dataset.numeric_stats.items()},
    }
    with MODEL_PATH.open("w", encoding="utf-8") as f:
        json.dump(model_payload, f)

    metrics = {
        "train": evaluate(weights, train_set),
        "test": evaluate(weights, test_set),
    }

    metrics_path = MODEL_DIR / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = train_and_save_model()
    print(json.dumps(metrics, indent=2))
