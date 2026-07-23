#!/usr/bin/env python3
"""Create the minimal dataset layout required by the RAVEN pipeline."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def ensure_link(destination: Path, source: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing link points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing asset: {destination}")
    destination.symlink_to(source, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--labels", required=True, help="Official ready-to-use labels.csv")
    parser.add_argument("--freespace", required=True, help="Official radar_Freespace directory")
    args = parser.parse_args()
    root = Path(args.dataset_root).expanduser().resolve()
    labels = Path(args.labels).expanduser().resolve()
    freespace = Path(args.freespace).expanduser().resolve()
    if not labels.is_file() or not freespace.is_dir():
        raise FileNotFoundError("The labels file or freespace directory is missing.")
    root.mkdir(parents=True, exist_ok=True)
    (root / "ADC_Data").mkdir(exist_ok=True)
    destination = root / "labels.csv"
    if destination.exists():
        if destination.read_bytes() != labels.read_bytes():
            raise RuntimeError(f"Existing labels differ from {labels}")
    else:
        shutil.copy2(labels, destination)
    ensure_link(root / "radar_Freespace", freespace)
    print(f"Dataset root prepared: {root}")


if __name__ == "__main__":
    main()
