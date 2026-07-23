#!/usr/bin/env python3
"""Run one real RADIal frame through RAVEN, loss computation, and backward."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from raven_repro.config import load_config
from raven_repro.data import RADIalADCDataset, RangeAngleEncoder, radial_collate
from raven_repro.models import RAVENADC
from raven_repro.runtime import resolve_device
from raven_repro.training import build_segmentation_loss
from raven_repro.training.losses import compute_multitask_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/raven_raw_adc_sequence.json")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device)
    encoder = RangeAngleEncoder(
        config["dataset"]["geometry"], config["dataset"]["statistics"]
    )
    dataset = RADIalADCDataset(
        args.dataset_root,
        encoder.encode,
        include_difficult=config["dataset"].get("include_difficult", True),
    )
    batch = radial_collate([dataset[0]])
    adc = batch["adc"].to(device)
    detection_target = batch["detection_target"].to(device)
    segmentation_target = batch["segmentation_target"].float().unsqueeze(1).to(device)

    model = RAVENADC(config["model"]).to(device).train()
    outputs = model(adc)
    criterion = build_segmentation_loss(config["loss"]["segmentation"])
    losses = compute_multitask_loss(
        outputs,
        detection_target,
        segmentation_target,
        criterion,
        config["loss"],
    )
    if not torch.isfinite(losses["total"]):
        raise FloatingPointError(f"Non-finite smoke-test loss: {losses}")
    losses["total"].backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise FloatingPointError("The smoke test produced a non-finite gradient.")
    print(f"samples={len(dataset)} sample_id={int(batch['sample_ids'][0])}")
    print(f"adc={tuple(adc.shape)} {adc.dtype}")
    print(
        f"Detection={tuple(outputs['Detection'].shape)} "
        f"Segmentation={tuple(outputs['Segmentation'].shape)}"
    )
    print(f"loss={float(losses['total'].detach()):.6f}; backward=finite")


if __name__ == "__main__":
    main()
