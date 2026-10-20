"""Image/mask pairs for training, with label-aware augmentation."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dentex.labels import hflip_lookup

SIZE = 512


def resize_pair(image: Image.Image, mask: Image.Image | None, size: int = SIZE):
    """Bilinear for the X-ray, nearest-neighbour for the label map (labels must not be interpolated)."""
    image = image.convert("L").resize((size, size), Image.BILINEAR)
    if mask is not None:
        mask = mask.resize((size, size), Image.NEAREST)
    return image, mask


class SegmentationDataset(Dataset):
    """Reads ``<root>/images/*.png`` and ``<root>/masks/*.png`` (same file names, already 512x512)."""

    def __init__(self, root: str | Path, task: str | None, names: list[str] | None = None,
                 augment: bool = False, hflip_p: float = 0.5, noise_p: float = 0.3):
        """``task`` selects the left/right label remapping used by flip augmentation;
        ``None`` (e.g. the {background, tooth, border} maps) means labels are flip-invariant."""
        root = Path(root)
        self.image_dir, self.mask_dir = root / "images", root / "masks"
        self.names = names or sorted(p.name for p in self.image_dir.glob("*.png"))
        self.augment, self.hflip_p, self.noise_p = augment, hflip_p, noise_p
        table = np.arange(256, dtype=np.int64)
        if task is not None:
            remap = hflip_lookup(task)
            table[: len(remap)] = remap
        self.flip_table = table

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        image = np.asarray(Image.open(self.image_dir / name).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(self.mask_dir / name), dtype=np.int64)

        if self.augment:
            if random.random() < self.hflip_p:
                image = image[:, ::-1]
                mask = self.flip_table[mask[:, ::-1]]       # left/right quadrants swap
            if random.random() < self.noise_p:           # salt-and-pepper, as in Helli & Hamamci (2022)
                noise = np.random.rand(*image.shape)
                image = image.copy()
                image[noise < 0.01] = 0.0
                image[noise > 0.99] = 1.0

        return {"name": name,
                "image": torch.from_numpy(np.ascontiguousarray(image))[None],
                "mask": torch.from_numpy(np.ascontiguousarray(mask))}


def split_names(names: list[str], val_fraction: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic train/validation split (80/20 for the U-Net experiments)."""
    rng = random.Random(seed)
    names = sorted(names)
    rng.shuffle(names)
    n_val = int(round(len(names) * val_fraction))
    return names[n_val:], names[:n_val]
