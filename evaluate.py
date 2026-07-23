#!/usr/bin/env python3
"""Evaluate a RAVEN checkpoint on train, validation, or test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from raven_repro.config import load_config
from raven_repro.data import (
    RADIalADCDataset,
    RangeAngleEncoder,
    SplitBundle,
    create_dataloaders,
    create_split_bundle,
)
from raven_repro.evaluation.runner import run_full_evaluation, write_evaluation
from raven_repro.models import RAVENADC
from raven_repro.runtime import resolve_device, seed_everything
from raven_repro.training import build_segmentation_loss
from raven_repro.training.checkpoint import load_checkpoint


def parse_thresholds(value: str) -> list[float]:
    values = [float(item) for item in value.split(",")]
    if not values or any(item <= 0 or item >= 1 for item in values):
        raise argparse.ArgumentTypeError("Thresholds must be comma-separated values in (0,1).")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--confidence", type=parse_thresholds, default=parse_thresholds("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"))
    parser.add_argument("--iou", type=parse_thresholds, default=parse_thresholds("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_root:
        config["paths"]["dataset_root"] = str(Path(args.dataset_root).expanduser().resolve())
    seed_everything(int(config["seed"]))
    device = resolve_device(args.device)
    encoder = RangeAngleEncoder(
        config["dataset"]["geometry"], config["dataset"]["statistics"]
    )
    dataset = RADIalADCDataset(
        config["paths"]["dataset_root"],
        encoder.encode,
        include_difficult=config["dataset"].get("include_difficult", True),
    )
    state = load_checkpoint(args.checkpoint)
    split = SplitBundle(**state["split"]) if "split" in state else create_split_bundle(
        dataset, config["split"], int(config["seed"])
    )
    loaders = create_dataloaders(dataset, split, config["loader"], int(config["seed"]))
    model = RAVENADC(config["model"]).to(device)
    model.load_state_dict(state["model_state_dict"])
    criterion = build_segmentation_loss(config["loss"]["segmentation"])
    result = run_full_evaluation(
        model,
        loaders[args.split],
        encoder,
        criterion,
        config,
        device,
        args.confidence,
        args.iou,
    )
    result.update(
        {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_epoch": int(state.get("epoch", -1)),
            "split": args.split,
            "test_aliases_val": split.test_aliases_val,
        }
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / f"evaluation_{args.split}"
    write_evaluation(result, output_dir)
    print(f"Evaluation written to: {output_dir.resolve()}")
    print(f"Loss: {result['loss']:.6f}")
    print(f"Fixed detection: {result['fixed_detection']}")
    print(
        f"Segmentation mIoU full={result['segmentation_miou_full']:.6f}, "
        f"near={result['segmentation_miou_near_range']:.6f}"
    )


if __name__ == "__main__":
    main()
