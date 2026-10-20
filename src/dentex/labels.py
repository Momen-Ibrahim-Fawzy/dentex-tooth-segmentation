"""Label spaces of the three DENTEX tasks.

DENTEX annotations are hierarchical. Every box/polygon carries
    category_id_1  quadrant index 0..3  -> FDI quadrants 1..4
    category_id_2  tooth index 0..7     -> FDI tooth 1..8 inside the quadrant
    category_id_3  diagnosis 0..3       -> impacted, caries, periapical lesion, deep caries

Each task is trained as a separate semantic-segmentation problem, with label 0 = background:
    quadrant      4 classes   label = q + 1
    enumeration  32 classes   label = q * 8 + t + 1          (Q0T0 .. Q3T7)
    disease       4 classes   label = d + 1
"""
from __future__ import annotations

from dataclasses import dataclass

DIAGNOSES = ("impacted", "caries", "periapical_lesion", "deep_caries")


@dataclass(frozen=True)
class Task:
    name: str
    num_classes: int          # including background
    label_names: tuple[str, ...]

    def nnunet_labels(self) -> dict[str, int]:
        return {"background": 0, **{n: i + 1 for i, n in enumerate(self.label_names)}}


TASKS = {
    "quadrant": Task("quadrant", 5, tuple(str(q) for q in range(4))),
    "enumeration": Task("enumeration", 33, tuple(f"Q{q}T{t}" for q in range(4) for t in range(8))),
    "disease": Task("disease", 5, DIAGNOSES),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"unknown task '{name}', choose from {sorted(TASKS)}")
    return TASKS[name]


def encode(task: str, quadrant: int, tooth: int | None = None, disease: int | None = None) -> int:
    """Annotation categories -> integer label for the given task."""
    if task == "quadrant":
        return quadrant + 1
    if task == "enumeration":
        return quadrant * 8 + tooth + 1
    if task == "disease":
        return disease + 1
    raise KeyError(task)


def decode_enumeration(label: int) -> tuple[int, int]:
    """Enumeration label (1..32) -> (quadrant index 0..3, tooth index 0..7)."""
    if not 1 <= label <= 32:
        raise ValueError(label)
    return (label - 1) // 8, (label - 1) % 8


def fdi(quadrant: int, tooth: int) -> int:
    """FDI two-digit tooth number, e.g. quadrant 0 / tooth 0 -> 11."""
    return (quadrant + 1) * 10 + (tooth + 1)


# A horizontal flip of a panoramic X-ray swaps the patient's left and right side:
# FDI quadrant 1 <-> 2 and 3 <-> 4. Labels must be remapped, otherwise flip
# augmentation would teach the network the wrong quadrant.
_QUADRANT_SWAP = {0: 1, 1: 0, 2: 3, 3: 2}


def hflip_lookup(task: str) -> list[int]:
    """Lookup table new_label = table[old_label] to apply after a horizontal flip."""
    n = get_task(task).num_classes
    table = list(range(n))
    if task == "quadrant":
        for q, q2 in _QUADRANT_SWAP.items():
            table[q + 1] = q2 + 1
    elif task == "enumeration":
        for label in range(1, 33):
            q, t = decode_enumeration(label)
            table[label] = _QUADRANT_SWAP[q] * 8 + t + 1
    return table
