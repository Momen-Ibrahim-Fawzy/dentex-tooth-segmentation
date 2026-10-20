"""Mask refinement for tooth *instance* segmentation with a semantic network.

A semantic U-Net cannot tell two touching teeth apart: neighbouring teeth merge
into one blob and post-processing counts the wrong number of teeth (for example
13 teeth on an X-ray with far more). Two refinements of the training masks fix this:

1. ``separate_teeth``  every tooth is grown by ``gap`` pixels; wherever it meets an
   already-drawn tooth, the (also grown) intersection is cut out of the mask.
   The ground truth then contains a clear gap between neighbouring teeth.
2. ``add_border_class``  the mask is eroded slightly and dilated by
   ``border_size``; the band between the two is labelled as a separate *border*
   class. The network learns to predict borders explicitly, and at inference the
   tooth cores (tooth minus border) are separate components that can be counted
   and grown back to full teeth (see :mod:`dentex.postprocess.instances`).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BACKGROUND, TOOTH, BORDER = 0, 1, 2


def _square(size: int) -> np.ndarray:
    return np.ones((2 * size + 1, 2 * size + 1), dtype=bool)


def separate_teeth(instances: np.ndarray, labels: np.ndarray | None = None, gap: int = 6) -> np.ndarray:
    """instances: (N,H,W) bool masks in drawing order. Returns an (H,W) label map with gaps.

    Mirrors drawing the polygons one by one: tooth k is grown by ``gap``, its
    intersection with every previously drawn tooth is grown by ``gap`` again and
    erased from both. ``labels`` gives the value for each tooth (default 1..N).
    """
    if instances.size == 0:
        return np.zeros(instances.shape[1:], dtype=np.int32)
    labels = np.arange(1, len(instances) + 1) if labels is None else np.asarray(labels)
    grow = _square(gap) if gap > 0 else None
    out = np.zeros(instances.shape[1:], dtype=np.int32)
    drawn = np.zeros(instances.shape[1:], dtype=bool)
    for inst, value in zip(instances, labels):
        current = ndimage.binary_dilation(inst, grow) if grow is not None else inst.copy()
        overlap = current & drawn
        if overlap.any():
            if grow is not None:
                overlap = ndimage.binary_dilation(overlap, grow)
            out[overlap] = 0
            current &= ~overlap
        out[current] = value
        drawn |= current
    return out


def remove_overlaps(instances: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """Strictest variant: only pixels claimed by two or more teeth become background (no gap)."""
    if instances.size == 0:
        return np.zeros(instances.shape[1:], dtype=np.int32)
    labels = np.arange(1, len(instances) + 1) if labels is None else np.asarray(labels)
    count = instances.sum(axis=0)
    out = np.zeros(instances.shape[1:], dtype=np.int32)
    for inst, value in zip(instances, labels):
        out[inst & (count == 1)] = value
    return out


def add_border_class(mask: np.ndarray, border_size: int = 5, n_erosions: int = 1) -> np.ndarray:
    """mask: (H,W), any value > 0 is tooth. Returns {0 background, 1 tooth, 2 border}.

    Erode by ``n_erosions`` (3x3), then dilate the eroded mask with a
    (2*border_size+1)^2 square: eroded pixels are tooth, the remaining dilated
    band (inside and outside the original outline) is border.
    """
    tooth = np.asarray(mask) > 0
    eroded = ndimage.binary_erosion(tooth, np.ones((3, 3), bool), iterations=n_erosions) if n_erosions else tooth
    dilated = ndimage.binary_dilation(eroded, _square(border_size))
    out = np.zeros(tooth.shape, dtype=np.uint8)
    out[dilated] = BORDER
    out[eroded] = TOOTH
    return out
