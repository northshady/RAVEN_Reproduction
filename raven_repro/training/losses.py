"""RADIal multitask losses used by RAVEN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "sum") -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.where(target == 1.0, prediction, 1.0 - prediction)
        loss = -((1.0 - probability) ** self.gamma) * torch.log(probability + 1e-6)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class JaccardLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        target = target.to(probabilities.dtype)
        dimensions = tuple(range(1, probabilities.ndim))
        intersection = (probabilities * target).sum(dim=dimensions)
        union = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions) - intersection
        return (1.0 - (intersection + self.smooth) / (union + self.smooth)).mean()


def build_segmentation_loss(name: str) -> nn.Module:
    normalized = name.replace("_", "").lower()
    if normalized in {"bce", "bcewithlogits", "bcewithlogitsloss"}:
        return nn.BCEWithLogitsLoss(reduction="mean")
    if normalized in {"jaccard", "jaccardloss", "iou", "iouloss"}:
        return JaccardLoss()
    raise ValueError(f"Unsupported segmentation loss: {name!r}")


def detection_losses(
    prediction: torch.Tensor, target: torch.Tensor, regression_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    classification = FocalLoss(gamma=2.0, reduction="sum")(
        prediction[:, 0].flatten(), target[:, 0].flatten()
    )
    mask = target[:, 0].unsqueeze(1)
    regression_prediction = prediction[:, 1:] * mask
    regression_target = target[:, 1:]
    normalized = regression_name.replace("_", "").lower()
    if normalized in {"smoothl1", "smoothl1loss", "huber"}:
        regression = F.smooth_l1_loss(
            regression_prediction, regression_target, reduction="sum"
        )
    elif normalized in {"l1", "l1loss"}:
        regression = F.l1_loss(regression_prediction, regression_target, reduction="sum")
    else:
        raise ValueError(f"Unsupported regression loss: {regression_name!r}")
    positives = mask.sum()
    if positives.item() > 0:
        regression = regression / positives
    return classification, regression


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    detection_target: torch.Tensor,
    segmentation_target: torch.Tensor,
    segmentation_criterion: nn.Module,
    config: dict,
) -> dict[str, torch.Tensor]:
    classification, regression = detection_losses(
        outputs["Detection"], detection_target, config["regression"]
    )
    segmentation = segmentation_criterion(
        outputs["Segmentation"], segmentation_target
    ) * outputs["Segmentation"].shape[0]
    weights = config["weights"]
    weighted_classification = classification * float(weights[0])
    weighted_regression = regression * float(weights[1])
    weighted_segmentation = segmentation * float(weights[2])
    total = weighted_classification + weighted_regression + weighted_segmentation
    return {
        "total": total,
        "classification": classification,
        "regression": regression,
        "segmentation": segmentation,
        "weighted_classification": weighted_classification,
        "weighted_regression": weighted_regression,
        "weighted_segmentation": weighted_segmentation,
    }
