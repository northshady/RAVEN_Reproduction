"""RAVEN full evaluation using the released FFTRadNet/RADIal protocol.

This module follows ``RADIal-main/FFTRadNet/utils/metrics.py``:
confidence thresholds 0.1..0.9, NMS IoU 0.05, matching IoU 0.5,
range 5..100 m, mean precision/recall across thresholds, and near-range
freespace mIoU over the first 124 rows.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def ra_to_cartesian_box(data: np.ndarray) -> np.ndarray:
    boxes = []
    for row in data:
        x = np.sin(np.radians(row[1])) * row[0]
        y = np.cos(np.radians(row[1])) * row[0]
        boxes.append(
            [x - 0.9, y, x + 0.9, y, x + 0.9, y + 4.0, x - 0.9, y + 4.0]
        )
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 8)


def bbox_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    first = Polygon(np.asarray(box).reshape(4, 2))
    first_area = first.area
    values = np.zeros(len(boxes), dtype=np.float64)
    for index, candidate in enumerate(boxes):
        second = Polygon(np.asarray(candidate).reshape(4, 2))
        intersection = first.intersection(second).area
        values[index] = intersection / (
            first_area + second.area - intersection
        )
    return values


def perform_nms(
    confidence: np.ndarray, boxes: np.ndarray, threshold: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(confidence)[::-1]
    boxes = boxes[order]
    confidence = confidence[order]
    index = 0
    while index < len(boxes):
        overlap = np.zeros(len(boxes), dtype=np.float64)
        if index + 1 < len(boxes):
            overlap[index + 1 :] = bbox_iou(boxes[index], boxes[index + 1 :])
        keep = overlap < threshold
        boxes = boxes[keep]
        confidence = confidence[keep]
        index += 1
    return confidence, boxes


def process_predictions(
    predictions: np.ndarray,
    confidence_threshold: float,
    nms_threshold: float = 0.05,
) -> np.ndarray:
    if len(predictions) == 0:
        return np.empty((0, 9), dtype=np.float64)
    boxes = ra_to_cartesian_box(predictions)
    confidence = predictions[:, -1]
    valid = confidence > confidence_threshold
    confidence, boxes = perform_nms(
        confidence[valid], boxes[valid], nms_threshold
    )
    return np.hstack((confidence[:, None], boxes))


def full_detection_metrics(
    predictions: list[np.ndarray],
    labels: list[np.ndarray],
    *,
    range_min: float = 5.0,
    range_max: float = 100.0,
    iou_threshold: float = 0.5,
) -> dict:
    threshold_rows = []
    for threshold in np.arange(0.1, 0.96, 0.1):
        true_positive = false_positive = false_negative = 0
        range_error = angle_error = 0.0
        matched_objects = 0
        for frame_predictions, frame_labels in zip(predictions, labels):
            objects = process_predictions(frame_predictions, threshold)
            if len(objects):
                distance = (objects[:, 2] + objects[:, 4]) / 2
                objects = objects[
                    (distance >= range_min) & (distance <= range_max)
                ]
            target = np.asarray(frame_labels)
            if len(target):
                target = target[
                    (target[:, 0] >= range_min) & (target[:, 0] <= range_max)
                ]
            target_boxes = ra_to_cartesian_box(target)

            if len(target_boxes) and len(objects):
                used = np.zeros(len(target_boxes), dtype=bool)
                for prediction in objects:
                    overlaps = bbox_iou(prediction[1:], target_boxes)
                    matches = np.where(overlaps >= iou_threshold)[0]
                    if len(matches):
                        true_positive += 1
                        used[matches] = True
                        # Preserve the released FFTRadNet full-evaluation
                        # regression-error calculation exactly.
                        range_error += np.sum(
                            np.abs(target_boxes[matches, -2] - prediction[-2])
                        )
                        angle_error += np.sum(
                            np.abs(target_boxes[matches, -1] - prediction[-1])
                        )
                        matched_objects += len(matches)
                    else:
                        false_positive += 1
                false_negative += int((~used).sum())
            elif not len(target_boxes):
                false_positive += len(objects)
            elif not len(objects):
                false_negative += len(target_boxes)

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive
            else 0.0
        )
        threshold_rows.append(
            {
                "confidence": float(round(threshold, 1)),
                "precision": float(precision),
                "recall": float(recall),
                "range_error": (
                    float(range_error / matched_objects)
                    if matched_objects
                    else None
                ),
                "angle_error": (
                    float(angle_error / matched_objects)
                    if matched_objects
                    else None
                ),
                "tp": int(true_positive),
                "fp": int(false_positive),
                "fn": int(false_negative),
            }
        )

    mean_precision = float(np.mean([row["precision"] for row in threshold_rows]))
    mean_recall = float(np.mean([row["recall"] for row in threshold_rows]))
    f1 = (
        2 * mean_precision * mean_recall / (mean_precision + mean_recall)
        if mean_precision + mean_recall
        else 0.0
    )
    valid_range = [
        row["range_error"] for row in threshold_rows if row["range_error"] is not None
    ]
    valid_angle = [
        row["angle_error"] for row in threshold_rows if row["angle_error"] is not None
    ]
    return {
        "mAP": mean_precision,
        "mAR": mean_recall,
        "F1": float(f1),
        "RE": float(np.mean(valid_range)) if valid_range else None,
        "AE": float(np.mean(valid_angle)) if valid_angle else None,
        "confidence_thresholds": threshold_rows,
        "protocol": {
            "source": "RADIal-main/FFTRadNet/utils/metrics.py::GetFullMetrics",
            "confidence_thresholds": [round(value, 1) for value in np.arange(0.1, 0.96, 0.1)],
            "nms_iou": 0.05,
            "matching_iou": iou_threshold,
            "range_m": [range_min, range_max],
            "allows_multiple_predictions_to_match_the_same_ground_truth": True,
        },
    }


def near_range_miou(
    predictions: list[np.ndarray], labels: list[np.ndarray], rows: int = 124
) -> float:
    values = []
    for prediction, target in zip(predictions, labels):
        predicted = np.asarray(prediction)[:rows].reshape(-1) >= 0.5
        expected = np.asarray(target)[:rows].reshape(-1).astype(bool)
        intersection = np.logical_and(predicted, expected).sum()
        union = expected.sum() + predicted.sum() - intersection
        values.append(float(intersection / union) if union else 1.0)
    return float(np.mean(values))
