"""Plain YAML + dataclass config loader (no Hydra/OmegaConf dependency).

`configs/base.yaml` holds shared defaults; `configs/pilot.yaml` /
`configs/full.yaml` only override scale-related fields on top of it. Every
script calls `load_config(scale=...)` and gets back the same `ExperimentConfig`
shape regardless of scale -- scaling up is a config change, not a code change.
"""

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from isplit.config.schema import ExperimentConfig


def _merge_into(instance: Any, overrides: dict) -> Any:
    for key, value in overrides.items():
        if not hasattr(instance, key):
            raise ValueError(f"unknown config field: {key!r} on {type(instance).__name__}")
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_into(current, value)
        else:
            setattr(instance, key, value)
    return instance


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(scale: str = "pilot", config_dir: str | Path = "configs") -> ExperimentConfig:
    config_dir = Path(config_dir)
    cfg = ExperimentConfig()

    base_overrides = _load_yaml(config_dir / "base.yaml")
    _merge_into(cfg, base_overrides)

    scale_overrides = _load_yaml(config_dir / f"{scale}.yaml")
    _merge_into(cfg, scale_overrides)

    cfg.scale = scale
    return cfg
