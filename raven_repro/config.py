"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config = deepcopy(config)
    for key in ("dataset_root", "output_dir"):
        value = os.path.expandvars(os.path.expanduser(config["paths"][key]))
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = path.parent.parent / candidate
        config["paths"][key] = str(candidate.resolve())
    validate_config(config)
    config["config_path"] = str(path)
    return config


def validate_config(config: dict) -> None:
    required = {
        "experiment_name",
        "seed",
        "paths",
        "preprocessing",
        "model",
        "dataset",
        "split",
        "loader",
        "loss",
        "optimizer",
        "scheduler",
        "training",
        "evaluation",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(f"Configuration is missing sections: {missing}")
    if int(config["training"]["epochs"]) <= 0:
        raise ValueError("training.epochs must be positive.")
    if float(config["optimizer"]["lr"]) <= 0:
        raise ValueError("optimizer.lr must be positive.")


def save_config(config: dict, path: str | Path) -> None:
    serializable = {key: value for key, value in config.items() if key != "config_path"}
    Path(path).write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
