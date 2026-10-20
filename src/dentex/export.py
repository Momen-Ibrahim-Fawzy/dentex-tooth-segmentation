"""Write predicted boxes as COCO detection results for the three DENTEX levels.

Boxes are predicted on 512x512 inputs and scaled back to each X-ray's original
size. The output follows the COCO results format extended with the
hierarchical DENTEX categories, so AP / AP50 / AP75 / AR can be computed per
level with standard COCO tooling:

    {"image_id": 12, "bbox": [x, y, w, h], "score": 0.93,
     "category_id_1": 2, "category_id_2": 5, "category_id_3": 1}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dentex.labels import decode_enumeration
from dentex.postprocess.boxes import Box


def _iou(a: Box, b: Box) -> float:
    ax0, ay0, ax1, ay1 = a.xyxy()
    bx0, by0, bx1, by1 = b.xyxy()
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def assign_teeth(disease_boxes: list[Box], tooth_boxes: list[Box]) -> list[tuple[Box, Box | None]]:
    """Attach every diseased region to the enumerated tooth it overlaps most."""
    out = []
    for d in disease_boxes:
        best = max(tooth_boxes, key=lambda t: _iou(d, t), default=None)
        out.append((d, best if best is not None and _iou(d, best) > 0 else None))
    return out


def to_records(image_id: int, original_size: tuple[int, int], *, quadrant: list[Box] | None = None,
               enumeration: list[Box] | None = None, disease: list[Box] | None = None,
               input_size: int = 512) -> list[dict]:
    """original_size = (width, height) of the X-ray before resizing."""
    sx, sy = original_size[0] / input_size, original_size[1] / input_size
    rec = []

    def add(box: Box, **cats):
        b = box.scaled(sx, sy)
        rec.append({"image_id": image_id, "bbox": [round(v, 2) for v in (b.x, b.y, b.w, b.h)],
                    "score": round(b.score, 4), **cats})

    for b in quadrant or []:
        add(b, category_id_1=b.label - 1)
    for b in enumeration or []:
        q, t = decode_enumeration(b.label)
        add(b, category_id_1=q, category_id_2=t)
    if disease:
        for d, tooth in assign_teeth(disease, enumeration or []):
            if tooth is None:
                continue  # a lesion that does not overlap any detected tooth cannot be enumerated
            q, t = decode_enumeration(tooth.label)
            add(d, category_id_1=q, category_id_2=t, category_id_3=d.label - 1)
    return rec


def write_predictions(records: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else o))
