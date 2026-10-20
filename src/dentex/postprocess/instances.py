"""From semantic predictions to individual teeth.

Two strategies:

* ``instances_from_border_prediction``: the network predicts {background,
  tooth, border}. Tooth pixels that are not border form separated "cores", one
  per tooth. Each core is labelled, small fragments are dropped, and every
  remaining tooth pixel is assigned to its nearest core.
* ``instances_from_binary``: for a plain tooth/background mask (the original
  U-Net output), touching teeth are separated by morphological opening before
  labelling, as in Helli & Hamamci (2022).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from dentex.data.refine import BORDER, TOOTH


def _grow_to_region(seeds: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Assign every pixel of ``region`` to its nearest labelled seed."""
    if seeds.max() == 0:
        return seeds
    _, (iy, ix) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
    grown = seeds[iy, ix]
    return np.where(region, grown, 0)


def _drop_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    if labels.max() == 0:
        return labels
    areas = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, labels.max() + 1))
    keep = np.zeros(labels.max() + 1, dtype=bool)
    keep[1:] = np.asarray(areas) >= min_area
    relabelled, _ = ndimage.label(keep[labels])
    return relabelled


def instances_from_border_prediction(pred: np.ndarray, min_core_area: int = 30) -> np.ndarray:
    """pred: (H,W) in {0 bg, 1 tooth, 2 border}. Returns an (H,W) instance map 0..N."""
    cores = pred == TOOTH
    cores = ndimage.binary_opening(cores, iterations=1)
    seeds, _ = ndimage.label(cores)
    seeds = _drop_small(seeds, min_core_area)
    return _grow_to_region(seeds, (pred == TOOTH) | (pred == BORDER))


def instances_from_binary(mask: np.ndarray, opening_iterations: int = 2, min_area: int = 50) -> np.ndarray:
    """mask: (H,W) bool tooth mask. Returns an (H,W) instance map 0..N."""
    mask = mask.astype(bool)
    separated = ndimage.binary_opening(mask, structure=np.ones((3, 3)), iterations=opening_iterations)
    seeds, _ = ndimage.label(separated)
    seeds = _drop_small(seeds, min_area)
    return _grow_to_region(seeds, mask)


def count_teeth(instance_map: np.ndarray) -> int:
    return int(len(np.unique(instance_map)) - (1 if (instance_map == 0).any() else 0))
