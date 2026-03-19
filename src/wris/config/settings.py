from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataConfig:
    url: str
    raw_path: Path


@dataclass
class ModelConfig:
    target: str
    test_size: float
    random_state: int
    k_neighbors: int


@dataclass
class ArtifactsConfig:
    model_path: Path
    metrics_path: Path


@dataclass
class Settings:
    data: DataConfig
    model: ModelConfig
    artifacts: ArtifactsConfig


def load_settings(config_path: str | Path = "configs.json") -> Settings:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return Settings(
        data=DataConfig(url=raw["data"]["url"], raw_path=Path(raw["data"]["raw_path"])),
        model=ModelConfig(
            target=raw["model"]["target"],
            test_size=float(raw["model"]["test_size"]),
            random_state=int(raw["model"]["random_state"]),
            k_neighbors=int(raw["model"]["k_neighbors"]),
        ),
        artifacts=ArtifactsConfig(
            model_path=Path(raw["artifacts"]["model_path"]),
            metrics_path=Path(raw["artifacts"]["metrics_path"]),
        ),
    )
