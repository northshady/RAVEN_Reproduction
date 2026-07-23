#!/usr/bin/env python3
"""Run the RAVEN full evaluation using the released RADIal protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from raven_repro.config import load_config
from raven_repro.data import (
    RADIalADCDataset,
    RangeAngleEncoder,
    SplitBundle,
    create_dataloaders,
)
from raven_repro.evaluation.raven_full import (
    full_detection_metrics,
    near_range_miou,
)
from raven_repro.models import RAVENADC
from raven_repro.runtime import resolve_device, seed_everything
from raven_repro.training.checkpoint import load_checkpoint


@torch.inference_mode()
def collect(model, loader, encoder, device):
    model.eval()
    detections = []
    object_labels = []
    segmentation = []
    segmentation_labels = []
    for batch in tqdm(loader, desc="RAVEN full evaluation"):
        adc = batch["adc"].to(device=device, dtype=torch.complex64, non_blocking=True)
        outputs = model(adc)
        detection_maps = outputs["Detection"].float().cpu().numpy()
        segmentation_maps = torch.sigmoid(outputs["Segmentation"]).float().cpu().numpy()
        targets = batch["segmentation_target"].numpy()
        for index in range(adc.shape[0]):
            # Released run_FullEvaluation decodes at 0.05 before the sweep.
            detections.append(encoder.decode(detection_maps[index], 0.05))
            object_labels.append(np.asarray(batch["labels"][index]))
            segmentation.append(segmentation_maps[index, 0])
            segmentation_labels.append(targets[index])
    return detections, object_labels, segmentation, segmentation_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["seed"]))
    device = resolve_device(args.device)
    encoder = RangeAngleEncoder(
        config["dataset"]["geometry"], config["dataset"]["statistics"]
    )
    dataset = RADIalADCDataset(
        args.dataset_root,
        encoder.encode,
        include_difficult=config["dataset"].get("include_difficult", True),
    )
    state = load_checkpoint(args.checkpoint)
    split = SplitBundle(**state["split"])
    loaders = create_dataloaders(dataset, split, config["loader"], int(config["seed"]))
    model = RAVENADC(config["model"]).to(device)
    model.load_state_dict(state["model_state_dict"])

    detections, labels, segmentation, segmentation_labels = collect(
        model, loaders[args.split], encoder, device
    )
    result = full_detection_metrics(
        detections,
        labels,
        range_min=5,
        range_max=100,
        iou_threshold=0.5,
    )
    result.update(
        {
            "mIoU": near_range_miou(segmentation, segmentation_labels, rows=124),
            "sample_count": len(loaders[args.split].dataset),
            "split": args.split,
            "test_aliases_val": split.test_aliases_val,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": int(state["epoch"]) + 1,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("mIoU", "F1", "mAP", "mAR", "RE", "AE")}, indent=2))
    print(f"Written to {output.resolve()}")


if __name__ == "__main__":
    main()
