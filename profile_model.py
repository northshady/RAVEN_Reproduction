#!/usr/bin/env python3
"""Print RAVEN parameter groups and analytical MAC estimates."""

from __future__ import annotations

import argparse

from raven_repro.config import load_config
from raven_repro.models import RAVENADC


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/raven_raw_adc_sequence.json")
    args = parser.parse_args()
    config = load_config(args.config)
    model = RAVENADC(config["model"])
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("Logical parameter groups:")
    for name, value in model.logical_parameter_counts().items():
        print(f"  {name}: {value:,}")
    print("Analytical MAC estimate:")
    for name, value in model.raven.analytical_profile().items():
        print(f"  {name}: {value:,}")


if __name__ == "__main__":
    main()
