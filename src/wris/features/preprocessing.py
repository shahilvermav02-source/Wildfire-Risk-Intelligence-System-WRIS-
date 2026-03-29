from __future__ import annotations

import math

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_ORDER = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MONTH_TO_IDX = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}
DAY_TO_IDX = {d: i + 1 for i, d in enumerate(DAY_ORDER)}

BASE_FEATURES = ["x", "y", "ffmc", "dmc", "dc", "isi", "temp", "rh", "wind", "rain"]



def _normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "")


def _canonical_row(row: dict[str, str | float | int]) -> dict[str, str | float | int]:
    return {_normalize_key(k): v for k, v in row.items()}


def _get_required_value(row: dict[str, str | float | int], key: str) -> str | float | int:
    canonical = _canonical_row(row)
    normalized = _normalize_key(key)
    if normalized not in canonical:
        raise KeyError(f"Missing required field '{key}' in input row. Available keys: {list(canonical.keys())}")
    return canonical[normalized]


def row_to_features(row: dict[str, str | float | int]) -> list[float]:
    month = str(_get_required_value(row, "month")).strip().lower()
    day = str(_get_required_value(row, "day")).strip().lower()
    month_idx = MONTH_TO_IDX.get(month, 1)
    day_idx = DAY_TO_IDX.get(day, 1)

    values: list[float] = []
    for name in BASE_FEATURES:
        values.append(float(_get_required_value(row, name)))

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
        y.append(float(_get_required_value(row, target)))
    return x, y
