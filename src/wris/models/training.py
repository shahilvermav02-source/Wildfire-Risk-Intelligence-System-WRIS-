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
    weighted: bool = True
    x_train: list[list[float]] | None = None
    y_train: list[float] | None = None
    feature_mins: list[float] | None = None
    feature_ranges: list[float] | None = None

    def fit(self, x: list[list[float]], y: list[float]) -> None:
        self.x_train = x
        self.y_train = y

        cols = len(x[0])
        mins: list[float] = []
        ranges: list[float] = []
        for c in range(cols):
            col_vals = [row[c] for row in x]
            mn = min(col_vals)
            mx = max(col_vals)
            mins.append(mn)
            ranges.append(mx - mn if mx > mn else 1.0)

        self.feature_mins = mins
        self.feature_ranges = ranges

    def _scale(self, row: list[float]) -> list[float]:
        if self.feature_mins is None or self.feature_ranges is None:
            raise RuntimeError("Model scaling parameters are not fitted.")
        return [
            (value - self.feature_mins[idx]) / self.feature_ranges[idx]
            for idx, value in enumerate(row)
        ]

    def predict(self, x: list[list[float]]) -> list[float]:
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Model is not fitted.")

        scaled_train = [self._scale(row) for row in self.x_train]

        preds: list[float] = []
        for row in x:
            scaled_row = self._scale(row)
            dists = []
            for idx, train_row in enumerate(scaled_train):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(scaled_row, train_row)))
                dists.append((dist, self.y_train[idx]))
            dists.sort(key=lambda item: item[0])
            k_nearest = dists[: max(1, self.k_neighbors)]

            if self.weighted:
                weights = [1.0 / (dist + 1e-9) for dist, _ in k_nearest]
                weighted_sum = sum(w * y for w, (_, y) in zip(weights, k_nearest))
                pred = weighted_sum / sum(weights)
            else:
                pred = sum(val for _, val in k_nearest) / len(k_nearest)
            preds.append(pred)

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


def _cross_validated_k(
    x_train: list[list[float]],
    y_train: list[float],
    random_state: int,
) -> int:
    n = len(x_train)
    max_k = min(15, max(3, n - 1))
    candidates = [k for k in range(1, max_k + 1, 2)]
    if not candidates:
        return 1

    fold_count = 3 if n >= 9 else 2
    idx = list(range(n))
    rng = random.Random(random_state)
    rng.shuffle(idx)

    best_k = 1
    best_rmse = float("inf")

    for k in candidates:
        fold_rmses: list[float] = []
        for fold in range(fold_count):
            valid_idx = [i for pos, i in enumerate(idx) if pos % fold_count == fold]
            train_idx = [i for i in idx if i not in valid_idx]
            if not valid_idx or not train_idx:
                continue

            x_tr = [x_train[i] for i in train_idx]
            y_tr = [y_train[i] for i in train_idx]
            x_val = [x_train[i] for i in valid_idx]
            y_val = [y_train[i] for i in valid_idx]

            model = KNNRegressor(k_neighbors=min(k, len(x_tr)), weighted=True)
            model.fit(x_tr, y_tr)
            pred = model.predict(x_val)
            fold_rmses.append(_metrics(y_val, pred)["rmse"])

        if fold_rmses:
            avg_rmse = sum(fold_rmses) / len(fold_rmses)
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_k = k

    return best_k


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

    tuned_k = _cross_validated_k(x_train, y_train, random_state=random_state)
    final_k = min(max(1, tuned_k), len(x_train))

    model = KNNRegressor(k_neighbors=final_k, weighted=True)
    model.fit(x_train, y_train)

    pred = model.predict(x_test)
    metrics = _metrics(y_test, pred)
    metrics["k_selected"] = float(final_k)
    metrics["k_config"] = float(k_neighbors)
    return model, metrics


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
