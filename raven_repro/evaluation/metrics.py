"""Detection and freespace metrics for RADIal range-angle predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def range_angle_to_boxes(values: np.ndarray) -> np.ndarray:
    """Convert ``[range, angle, ...]`` to axis-aligned 4 m x 1.8 m boxes."""

    if len(values) == 0:
        return np.empty((0, 4), dtype=np.float32)
    x = np.sin(np.deg2rad(values[:, 1])) * values[:, 0]
    y = np.cos(np.deg2rad(values[:, 1])) * values[:, 0]
    return np.stack((x - 0.9, y, x + 0.9, y + 4.0), axis=1).astype(np.float32)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)
    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(box_area + areas - intersection, 1e-12)


def nms(predictions: np.ndarray, iou_threshold: float = 0.05) -> np.ndarray:
    if len(predictions) == 0:
        return predictions.reshape(0, 3)
    order = np.argsort(predictions[:, 2])[::-1]
    boxes = range_angle_to_boxes(predictions)
    keep = []
    while len(order):
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        order = remaining[box_iou(boxes[current], boxes[remaining]) < iou_threshold]
    return predictions[keep]


def match_frame(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float,
    range_min: float,
    range_max: float,
    nms_threshold: float,
) -> tuple[int, int, int]:
    predictions = predictions[predictions[:, 2] >= confidence_threshold]
    predictions = predictions[
        (predictions[:, 0] >= range_min) & (predictions[:, 0] <= range_max)
    ]
    predictions = nms(predictions, nms_threshold)
    labels = labels[(labels[:, 0] >= range_min) & (labels[:, 0] <= range_max)]
    prediction_boxes = range_angle_to_boxes(predictions)
    label_boxes = range_angle_to_boxes(labels)
    used = np.zeros(len(label_boxes), dtype=bool)
    true_positive = 0
    false_positive = 0
    for prediction_box in prediction_boxes:
        overlap = box_iou(prediction_box, label_boxes)
        overlap[used] = -1.0
        if len(overlap) and overlap.max() >= iou_threshold:
            matched = int(overlap.argmax())
            used[matched] = True
            true_positive += 1
        else:
            false_positive += 1
    false_negative = int((~used).sum())
    return true_positive, false_positive, false_negative


@dataclass
class DetectionAccumulator:
    confidence_threshold: float = 0.2
    iou_threshold: float = 0.2
    range_min: float = 5.0
    range_max: float = 100.0
    nms_threshold: float = 0.05
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        values = match_frame(
            predictions,
            labels,
            self.confidence_threshold,
            self.iou_threshold,
            self.range_min,
            self.range_max,
            self.nms_threshold,
        )
        self.true_positive += values[0]
        self.false_positive += values[1]
        self.false_negative += values[2]

    def result(self) -> dict[str, float]:
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        precision = self.true_positive / precision_denominator if precision_denominator else 0.0
        recall = self.true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "fn": self.false_negative,
        }


@dataclass
class SegmentationAccumulator:
    threshold: float = 0.5
    max_rows: int | None = None

    def __post_init__(self):
        self.values: list[float] = []

    def update(self, probability: np.ndarray, target: np.ndarray) -> None:
        if self.max_rows is not None:
            probability = probability[: self.max_rows]
            target = target[: self.max_rows]
        prediction = probability >= self.threshold
        target = target.astype(bool)
        intersection = np.logical_and(prediction, target).sum()
        union = np.logical_or(prediction, target).sum()
        self.values.append(float(intersection / union) if union else 1.0)

    def result(self) -> float:
        return float(np.mean(self.values)) if self.values else 0.0


def sweep_detection_metrics(
    predictions: list[np.ndarray],
    labels: list[np.ndarray],
    confidence_thresholds: list[float],
    iou_thresholds: list[float],
    *,
    range_min: float,
    range_max: float,
    nms_threshold: float,
) -> list[dict]:
    rows = []
    for confidence in confidence_thresholds:
        for iou in iou_thresholds:
            metric = DetectionAccumulator(
                confidence_threshold=confidence,
                iou_threshold=iou,
                range_min=range_min,
                range_max=range_max,
                nms_threshold=nms_threshold,
            )
            for frame_prediction, frame_labels in zip(predictions, labels):
                metric.update(frame_prediction, frame_labels)
            result = metric.result()
            rows.append({"confidence": confidence, "iou": iou, **result})
    return rows
