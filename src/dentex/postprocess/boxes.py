"""Segmentation masks -> one bounding box per predicted label.

A predicted label (e.g. tooth 36, or "caries") often appears as one large
region plus a few stray fragments. For every label:

1. find its connected components and their bounding boxes;
2. cluster the component *areas* with DBSCAN;
3. keep the cluster with the largest total area and return the box enclosing it.

This drops small spurious fragments without a hand-tuned area threshold, while
keeping a tooth that was split into two comparable pieces in one box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Box:
    label: int
    x: float
    y: float
    w: float
    h: float
    score: float

    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.x + self.w, self.y + self.h

    def scaled(self, sx: float, sy: float) -> "Box":
        return Box(self.label, self.x * sx, self.y * sy, self.w * sx, self.h * sy, self.score)


def dbscan_1d(values: np.ndarray, eps: float, min_samples: int = 1) -> np.ndarray:
    """DBSCAN on scalar values. Returns cluster ids (-1 = noise) in the input order."""
    values = np.asarray(values, dtype=float)
    labels = np.full(len(values), -1, dtype=int)
    order = np.argsort(values)
    cluster, start = -1, 0
    sv = values[order]
    # in 1-D, density-connected sets are maximal runs whose consecutive gaps are <= eps
    while start < len(sv):
        end = start
        while end + 1 < len(sv) and sv[end + 1] - sv[end] <= eps:
            end += 1
        if end - start + 1 >= min_samples:
            cluster += 1
            labels[order[start:end + 1]] = cluster
        start = end + 1
    return labels


def component_boxes(mask: np.ndarray) -> list[tuple[slice, slice, int]]:
    labelled, n = ndimage.label(mask)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labelled), start=1):
        if sl is not None:
            out.append((sl[0], sl[1], int((labelled[sl] == i).sum())))
    return out


def _make_box(mask: np.ndarray, label: int, chosen: list, probability: np.ndarray | None) -> Box:
    y0 = min(c[0].start for c in chosen)
    y1 = max(c[0].stop for c in chosen)
    x0 = min(c[1].start for c in chosen)
    x1 = max(c[1].stop for c in chosen)
    score = 1.0
    if probability is not None:
        region = np.zeros_like(mask, dtype=bool)
        region[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        score = float(probability[region].mean()) if region.any() else 0.0
    return Box(label, float(x0), float(y0), float(x1 - x0), float(y1 - y0), score)


def label_boxes(mask: np.ndarray, label: int, eps_fraction: float = 0.5,
                probability: np.ndarray | None = None, per_component: bool = False) -> list[Box]:
    """Boxes for one label from a boolean mask; ``probability`` (H,W) gives the confidence score.

    ``per_component=False`` (quadrants, enumerated teeth: one object per label) returns one box
    around the dominant cluster. ``per_component=True`` (diagnoses: several lesions can share a
    label) returns one box per component of the dominant cluster.
    """
    comps = component_boxes(mask)
    if not comps:
        return []
    areas = np.array([c[2] for c in comps], dtype=float)
    clusters = dbscan_1d(areas, eps=max(1.0, eps_fraction * areas.max()))
    best = max(set(clusters.tolist()), key=lambda c: areas[clusters == c].sum())
    chosen = [comps[i] for i in np.flatnonzero(clusters == best)]
    if per_component:
        return [_make_box(mask, label, [c], probability) for c in chosen]
    return [_make_box(mask, label, chosen, probability)]


def boxes_from_labelmap(labelmap: np.ndarray, probabilities: np.ndarray | None = None,
                        eps_fraction: float = 0.5, per_component: bool = False) -> list[Box]:
    """labelmap: (H,W) ints; probabilities: optional (C,H,W) softmax output."""
    boxes = []
    for label in np.unique(labelmap):
        if label == 0:
            continue
        prob = probabilities[label] if probabilities is not None else None
        boxes += label_boxes(labelmap == label, int(label), eps_fraction, prob, per_component)
    return boxes
