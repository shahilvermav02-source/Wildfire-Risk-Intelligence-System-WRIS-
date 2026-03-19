from __future__ import annotations

import math

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_ORDER = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MONTH_TO_IDX = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
DAY_TO_IDX = {d: i + 1 for i, d in enumerate(DAY_ORDER)}

BASE_FEATURES = ["x", "y", "ffmc", "dmc", "dc", "isi", "temp", "rh", "wind", "rain"]


def row_to_features(row: dict[str, str | float | int]) -> list[float]:
    month = str(row["month"]).strip().lower()
    day = str(row["day"]).strip().lower()
    month_idx = MONTH_TO_IDX.get(month, 1)
    day_idx = DAY_TO_IDX.get(day, 1)

    values: list[float] = []
    for name in BASE_FEATURES:
        values.append(float(row[name]))

    values.extend(
        [
            math.sin(2 * math.pi * month_idx / 12),
            math.cos(2 * math.pi * month_idx / 12),
            math.sin(2 * math.pi * day_idx / 7),
            math.cos(2 * math.pi * day_idx / 7),
            float(month_idx),
            float(day_idx),
        ]
    )
    return values


def rows_to_matrix(rows: list[dict[str, str]], target: str) -> tuple[list[list[float]], list[float]]:
    x, y = [], []
    for row in rows:
        x.append(row_to_features(row))
        y.append(float(row[target]))
    return x, y
