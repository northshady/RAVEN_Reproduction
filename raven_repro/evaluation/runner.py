"""One-pass loss, prediction collection, and threshold-sweep evaluation."""

from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from raven_repro.evaluation.metrics import (
    DetectionAccumulator,
    SegmentationAccumulator,
    sweep_detection_metrics,
)
from raven_repro.training.losses import compute_multitask_loss


def _autocast(device, enabled):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.float16, enabled=enabled)


@torch.inference_mode()
def run_full_evaluation(
    model,
    loader,
    encoder,
    segmentation_criterion,
    config: dict,
    device: torch.device,
    confidence_thresholds: list[float],
    iou_thresholds: list[float],
) -> dict:
    model.eval()
    minimum_confidence = min(confidence_thresholds)
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    sample_ids: list[int] = []
    full_segmentation = SegmentationAccumulator(threshold=0.5, max_rows=None)
    near_segmentation = SegmentationAccumulator(
        threshold=0.5, max_rows=config["evaluation"].get("segmentation_rows", 124)
    )
    loss_total = 0.0
    amp = bool(config["training"].get("amp", False))

    for batch in tqdm(loader, desc="evaluate"):
        adc = batch["adc"].to(device=device, dtype=torch.complex64, non_blocking=True)
        detection_target = batch["detection_target"].to(device=device, dtype=torch.float32)
        segmentation_target = (
            batch["segmentation_target"].to(device=device, dtype=torch.float32).unsqueeze(1)
        )
        with _autocast(device, amp):
            outputs = model(adc)
            losses = compute_multitask_loss(
                outputs,
                detection_target,
                segmentation_target,
                segmentation_criterion,
                config["loss"],
            )
        loss_total += float(losses["total"]) * adc.shape[0]
        detection_output = outputs["Detection"].float().cpu().numpy()
        segmentation_output = torch.sigmoid(outputs["Segmentation"]).float().cpu().numpy()
        segmentation_labels = segmentation_target.cpu().numpy()
        for index in range(adc.shape[0]):
            predictions.append(encoder.decode(detection_output[index], minimum_confidence))
            labels.append(np.asarray(batch["labels"][index]))
            sample_ids.append(int(batch["sample_ids"][index]))
            full_segmentation.update(
                segmentation_output[index, 0], segmentation_labels[index, 0]
            )
            near_segmentation.update(
                segmentation_output[index, 0], segmentation_labels[index, 0]
            )

    metric_config = config["evaluation"]
    sweep = sweep_detection_metrics(
        predictions,
        labels,
        confidence_thresholds,
        iou_thresholds,
        range_min=float(metric_config["range_min"]),
        range_max=float(metric_config["range_max"]),
        nms_threshold=float(metric_config["nms_threshold"]),
    )
    fixed = DetectionAccumulator(
        confidence_threshold=float(metric_config["confidence_threshold"]),
        iou_threshold=float(metric_config["iou_threshold"]),
        range_min=float(metric_config["range_min"]),
        range_max=float(metric_config["range_max"]),
        nms_threshold=float(metric_config["nms_threshold"]),
    )
    for prediction, target in zip(predictions, labels):
        fixed.update(prediction, target)
    return {
        "sample_count": len(loader.dataset),
        "loss": loss_total / len(loader.dataset),
        "fixed_detection": fixed.result(),
        "segmentation_miou_full": full_segmentation.result(),
        "segmentation_miou_near_range": near_segmentation.result(),
        "detection_sweep": sweep,
        "sample_ids": sample_ids,
    }


def write_evaluation(result: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    rows = result["detection_sweep"]
    with (output_dir / "detection_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
