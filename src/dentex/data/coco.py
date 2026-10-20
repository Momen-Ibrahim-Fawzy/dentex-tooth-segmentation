"""DENTEX COCO-style annotations -> segmentation masks.

The three DENTEX training subsets ship as COCO-like JSON files:
    train_quadrant.json                      (693 X-rays)
    train_quadrant_enumeration.json          (634 X-rays)
    train_quadrant_enumeration_disease.json  (705 X-rays)
Each annotation has a polygon ``segmentation``, a ``bbox`` and the hierarchical
category ids described in :mod:`dentex.labels`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dentex.labels import encode


@dataclass
class Annotation:
    polygons: list[list[float]]
    bbox: tuple[float, float, float, float]
    quadrant: int
    tooth: int | None = None
    disease: int | None = None


@dataclass
class XRay:
    image_id: int
    file_name: str
    width: int
    height: int
    annotations: list[Annotation]


def load_dentex(json_path: str | Path) -> list[XRay]:
    data = json.loads(Path(json_path).read_text())
    by_image: dict[int, list[Annotation]] = defaultdict(list)
    for a in data["annotations"]:
        seg = a.get("segmentation") or []
        polygons = [p for p in seg if len(p) >= 6] if isinstance(seg, list) else []
        by_image[a["image_id"]].append(Annotation(
            polygons=polygons,
            bbox=tuple(a["bbox"]),
            quadrant=a["category_id_1"],
            tooth=a.get("category_id_2"),
            disease=a.get("category_id_3"),
        ))
    return [XRay(img["id"], img["file_name"], img["width"], img["height"], by_image.get(img["id"], []))
            for img in data["images"]]


def _draw(ann: Annotation, size: tuple[int, int], value: int, canvas: Image.Image | None = None) -> Image.Image:
    canvas = canvas or Image.new("I", size, 0)
    draw = ImageDraw.Draw(canvas)
    if ann.polygons:
        for poly in ann.polygons:
            draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=value)
    else:  # fall back to the box when no polygon is available
        x, y, w, h = ann.bbox
        draw.rectangle([x, y, x + w, y + h], fill=value)
    return canvas


def semantic_mask(xray: XRay, task: str) -> np.ndarray:
    """(H, W) uint8 label map for one task; later annotations overwrite earlier ones."""
    canvas = Image.new("I", (xray.width, xray.height), 0)
    for ann in xray.annotations:
        if task == "disease" and ann.disease is None:
            continue
        if task == "enumeration" and ann.tooth is None:
            continue
        _draw(ann, canvas.size, encode(task, ann.quadrant, ann.tooth, ann.disease), canvas)
    return np.asarray(canvas, dtype=np.int32).astype(np.uint8)


def instance_masks(xray: XRay) -> np.ndarray:
    """(N, H, W) boolean stack, one mask per annotated object (used to find overlaps)."""
    if not xray.annotations:
        return np.zeros((0, xray.height, xray.width), dtype=bool)
    return np.stack([np.asarray(_draw(a, (xray.width, xray.height), 1), dtype=np.int32) > 0
                     for a in xray.annotations])
