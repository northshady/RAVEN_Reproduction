"""Training and validation loops."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
from tqdm.auto import tqdm

from raven_repro.evaluation.metrics import DetectionAccumulator, SegmentationAccumulator

from .losses import compute_multitask_loss


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "adc": batch["adc"].to(device=device, dtype=torch.complex64, non_blocking=True),
        "detection_target": batch["detection_target"].to(
            device=device, dtype=torch.float32, non_blocking=True
        ),
        "segmentation_target": batch["segmentation_target"]
        .to(device=device, dtype=torch.float32, non_blocking=True)
        .unsqueeze(1),
        "labels": batch["labels"],
        "sample_ids": batch["sample_ids"],
    }


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


def _empty_loss_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "classification_loss": 0.0,
        "regression_loss": 0.0,
        "segmentation_loss": 0.0,
    }


def _accumulate(totals: dict, losses: dict, batch_size: int) -> None:
    totals["loss"] += float(losses["total"].detach()) * batch_size
    totals["classification_loss"] += float(
        losses["weighted_classification"].detach()
    ) * batch_size
    totals["regression_loss"] += float(losses["weighted_regression"].detach()) * batch_size
    totals["segmentation_loss"] += float(
        losses["weighted_segmentation"].detach()
    ) * batch_size


def _average(totals: dict, sample_count: int) -> dict[str, float]:
    return {key: value / sample_count for key, value in totals.items()}


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    segmentation_criterion,
    loss_config: dict,
    device: torch.device,
    *,
    amp: bool,
    gradient_clip: float | None,
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals = _empty_loss_totals()
    progress = tqdm(loader, desc=f"train {epoch:03d}", leave=False)
    for raw_batch in progress:
        batch = _move_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            outputs = model(batch["adc"])
            losses = compute_multitask_loss(
                outputs,
                batch["detection_target"],
                batch["segmentation_target"],
                segmentation_criterion,
                loss_config,
            )
        if not torch.isfinite(losses["total"]):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}.")
        scaler.scale(losses["total"]).backward()
        if gradient_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        scaler.step(optimizer)
        scaler.update()
        batch_size = batch["adc"].shape[0]
        _accumulate(totals, losses, batch_size)
        progress.set_postfix(loss=f"{float(losses['total'].detach()):.3f}")
    return _average(totals, len(loader.dataset))


@torch.inference_mode()
def evaluate_epoch(
    model,
    loader,
    encoder,
    segmentation_criterion,
    loss_config: dict,
    metric_config: dict,
    device: torch.device,
    *,
    amp: bool,
    compute_metrics: bool,
    description: str = "val",
) -> dict[str, float | None]:
    model.eval()
    totals = _empty_loss_totals()
    detection = DetectionAccumulator(
        confidence_threshold=float(metric_config["confidence_threshold"]),
        iou_threshold=float(metric_config["iou_threshold"]),
        range_min=float(metric_config["range_min"]),
        range_max=float(metric_config["range_max"]),
        nms_threshold=float(metric_config["nms_threshold"]),
    )
    segmentation = SegmentationAccumulator(
        threshold=float(metric_config.get("segmentation_threshold", 0.5)),
        max_rows=metric_config.get("segmentation_rows"),
    )
    for raw_batch in tqdm(loader, desc=description, leave=False):
        batch = _move_batch(raw_batch, device)
        with _autocast(device, amp):
            outputs = model(batch["adc"])
            losses = compute_multitask_loss(
                outputs,
                batch["detection_target"],
                batch["segmentation_target"],
                segmentation_criterion,
                loss_config,
            )
        batch_size = batch["adc"].shape[0]
        _accumulate(totals, losses, batch_size)
        if compute_metrics:
            detection_output = outputs["Detection"].float().cpu().numpy()
            segmentation_output = torch.sigmoid(outputs["Segmentation"]).float().cpu().numpy()
            segmentation_target = batch["segmentation_target"].cpu().numpy()
            for index in range(batch_size):
                decoded = encoder.decode(
                    detection_output[index],
                    threshold=float(metric_config["confidence_threshold"]),
                )
                detection.update(decoded, np.asarray(batch["labels"][index]))
                segmentation.update(
                    segmentation_output[index, 0], segmentation_target[index, 0]
                )
    result: dict[str, float | None] = _average(totals, len(loader.dataset))
    if compute_metrics:
        result.update(detection.result())
        result["miou"] = segmentation.result()
        result["balanced_score"] = (float(result["f1"]) + float(result["miou"])) / 2
    else:
        result.update(
            {"precision": None, "recall": None, "f1": None, "miou": None, "balanced_score": None}
        )
    return result
