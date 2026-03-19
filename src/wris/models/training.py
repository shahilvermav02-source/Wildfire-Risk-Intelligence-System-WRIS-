from __future__ import annotations

import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

from wris.features.preprocessing import rows_to_matrix


@dataclass
class KNNRegressor:
    k_neighbors: int
    x_train: list[list[float]] | None = None
    y_train: list[float] | None = None

    def fit(self, x: list[list[float]], y: list[float]) -> None:
        self.x_train = x
        self.y_train = y

    def predict(self, x: list[list[float]]) -> list[float]:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted.")

        preds: list[float] = []
        for row in x:
            dists = []
            for idx, train_row in enumerate(self.x_train):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(row, train_row)))
                dists.append((dist, self.y_train[idx]))
            dists.sort(key=lambda item: item[0])
            k_nearest = dists[: max(1, self.k_neighbors)]
            preds.append(sum(val for _, val in k_nearest) / len(k_nearest))
        return preds


def _train_test_split(
    x: list[list[float]], y: list[float], test_size: float, random_state: int
) -> tuple[list[list[float]], list[list[float]], list[float], list[float]]:
    idx = list(range(len(x)))
    rng = random.Random(random_state)
    rng.shuffle(idx)

    cut = max(1, int(len(idx) * (1 - test_size)))
    train_idx = idx[:cut]
    test_idx = idx[cut:]

    x_train = [x[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return x_train, x_test, y_train, y_test


def _metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    n = len(y_true)
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / max(1, n)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / max(1, n))

    mean_true = sum(y_true) / max(1, n)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_true) ** 2 for a in y_true)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_evaluate(
    rows: list[dict[str, str]],
    target: str,
    test_size: float,
    random_state: int,
    k_neighbors: int,
) -> tuple[KNNRegressor, dict[str, float]]:
    x, y = rows_to_matrix(rows, target=target)
    x_train, x_test, y_train, y_test = _train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    model = KNNRegressor(k_neighbors=k_neighbors)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    return model, _metrics(y_test, pred)


def save_artifacts(
    model: KNNRegressor,
    metrics: dict[str, float],
    model_path: str | Path,
    metrics_path: str | Path,
) -> None:
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with model_path.open("wb") as f:
        pickle.dump(model, f)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
