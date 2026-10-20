"""Segmentation metrics from a confusion matrix.

Reports the metrics used throughout the project: accuracy, recall (sensitivity),
specificity, positive / negative predictive value, IoU and F1 (= Dice), either
for one foreground class versus the rest or macro-averaged over classes.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    pred, target = np.asarray(pred).ravel(), np.asarray(target).ravel()
    valid = (target >= 0) & (target < num_classes)
    return np.bincount(num_classes * target[valid] + pred[valid],
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)


def binary_scores(tp: float, fp: float, fn: float, tn: float, eps: float = 1e-9) -> dict[str, float]:
    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "recall": tp / (tp + fn + eps),
        "specificity": tn / (tn + fp + eps),
        "ppv": tp / (tp + fp + eps),
        "npv": tn / (tn + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
    }


def foreground_scores(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Any non-background class counts as foreground (the 'tooth vs background' view)."""
    p, t = np.asarray(pred) > 0, np.asarray(target) > 0
    tp = float((p & t).sum())
    fp = float((p & ~t).sum())
    fn = float((~p & t).sum())
    tn = float((~p & ~t).sum())
    return binary_scores(tp, fp, fn, tn)


def per_class_scores(cm: np.ndarray, skip_background: bool = True) -> dict[str, dict[str, float]]:
    total = cm.sum()
    out = {}
    for c in range(1 if skip_background else 0, cm.shape[0]):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        tn = float(total - tp - fp - fn)
        if tp + fn == 0 and tp + fp == 0:
            continue  # class absent from both prediction and ground truth
        out[str(c)] = binary_scores(tp, fp, fn, tn)
    return out


def macro(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    if not scores:
        return {}
    keys = next(iter(scores.values())).keys()
    return {k: float(np.mean([s[k] for s in scores.values()])) for k in keys}


class SegmentationMeter:
    """Accumulates a confusion matrix over a validation set."""

    def __init__(self, num_classes: int):
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.num_classes = num_classes

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        self.cm += confusion_matrix(pred, target, self.num_classes)

    def summary(self) -> dict[str, float]:
        per_class = per_class_scores(self.cm)
        result = {f"macro_{k}": v for k, v in macro(per_class).items()}
        result["dice"] = result.get("macro_f1", float("nan"))
        return result
