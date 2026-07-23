"""Checkpoint serialization helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    # Reproduction checkpoints contain optimizer/history/config Python objects
    # and are expected to be files generated locally by this repository.
    return torch.load(path, map_location=map_location, weights_only=False)
