from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictRequest:
    x: float
    y: float
    month: str
    day: str
    ffmc: float
    dmc: float
    dc: float
    isi: float
    temp: float
    rh: float
    wind: float
    rain: float

    @classmethod
    def from_dict(cls, data: dict) -> "PredictRequest":
        required = [
            "x", "y", "month", "day", "ffmc", "dmc", "dc", "isi", "temp", "rh", "wind", "rain"
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Missing fields: {', '.join(missing)}")

        obj = cls(
            x=float(data["x"]),
            y=float(data["y"]),
            month=str(data["month"]),
            day=str(data["day"]),
            ffmc=float(data["ffmc"]),
            dmc=float(data["dmc"]),
            dc=float(data["dc"]),
            isi=float(data["isi"]),
            temp=float(data["temp"]),
            rh=float(data["rh"]),
            wind=float(data["wind"]),
            rain=float(data["rain"]),
        )

        if not (0 <= obj.rh <= 100):
            raise ValueError("rh must be between 0 and 100")
        if min(obj.x, obj.y, obj.wind, obj.rain) < 0:
            raise ValueError("x, y, wind, and rain must be non-negative")
        return obj

    def to_feature_dict(self) -> dict[str, float | str]:
        return self.__dict__.copy()
