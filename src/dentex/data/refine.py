"""Mask refinement for tooth *instance* segmentation with a semantic network.

A semantic U-Net cannot tell two touching teeth apart: neighbouring teeth merge
into one blob and post-processing counts the wrong number of teeth. Two
refinements of the training masks fix this:

1. ``remove_overlaps``  pixels claimed by two annotated teeth become background,
   so the ground truth itself contains a thin gap between neighbours.
2. ``add_border_class`` the boundary band of every tooth gets its own class.
   The network learns to predict borders explicitly; at inference, tooth cores
   (tooth minus border) are separated components that can be counted and then
   grown back to full teeth (see :mod:`dentex.postprocess.instances`).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BACKGROUND, TOOTH, BORDER = 0, 1, 2


def remove_overlaps(instances: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """instances: (N,H,W) bool. Returns an (H,W) label map where overlapping pixels are 0.

    ``labels`` gives the value written for each instance (default 1..N).
    """
    if instances.size == 0:
        return np.zeros(instances.shape[1:], dtype=np.uint8)
    labels = np.arange(1, len(instances) + 1) if labels is None else np.asarray(labels)
    count = instances.sum(axis=0)
    out = np.zeros(instances.shape[1:], dtype=np.int32)
    for inst, value in zip(instances, labels):
        out[inst & (count == 1)] = value
    return out


def add_border_class(instance_map: np.ndarray, width: int = 2) -> np.ndarray:
    """instance_map: (H,W) ints, 0 = background, k = instance k. Returns {0 bg, 1 tooth, 2 border}."""
    out = np.where(instance_map > 0, TOOTH, BACKGROUND).astype(np.uint8)
    structure = ndimage.generate_binary_structure(2, 1)
    for k in np.unique(instance_map):
        if k == 0:
            continue
        inst = instance_map == k
        eroded = ndimage.binary_erosion(inst, structure, iterations=width)
        out[inst & ~eroded] = BORDER
    return out


def multiclass_with_border(semantic: np.ndarray, border_mask: np.ndarray, border_label: int) -> np.ndarray:
    """Keeps task classes (quadrant / enumeration) but overwrites borders with an extra class."""
    out = semantic.astype(np.int32).copy()
    out[border_mask] = border_label
    return out
