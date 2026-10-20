"""Builds nnU-Net v2 raw datasets for nnSAM training.

nnSAM (Li et al., 2023) plugs a frozen SAM image encoder into nnU-Net: SAM and
nnU-Net encode the image in parallel, their embeddings are concatenated and
decoded by nnU-Net's decoder. It reuses nnU-Net v2's data format and CLI, so
the only preparation needed is the nnU-Net raw folder layout:

    nnUNet_raw/Dataset043_DENTEX   quadrant
    nnUNet_raw/Dataset044_DENTEX   enumeration
    nnUNet_raw/Dataset045_DENTEX   disease
        imagesTr/DENTEX_0001_0000.png   (channel suffix _0000)
        labelsTr/DENTEX_0001.png
        imagesTs/...
        dataset.json
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from dentex.labels import get_task

DATASET_IDS = {"quadrant": 43, "enumeration": 44, "disease": 45}


def dataset_name(task: str) -> str:
    return f"Dataset{DATASET_IDS[task]:03d}_DENTEX"


def build_raw_dataset(task: str, images_dir: str | Path, masks_dir: str | Path, nnunet_raw: str | Path,
                      test_images_dir: str | Path | None = None) -> Path:
    """Copies resized X-rays and label maps into nnU-Net's naming scheme and writes dataset.json."""
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    root = Path(nnunet_raw) / dataset_name(task)
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in images_dir.glob("*.png"))
    for i, name in enumerate(names, start=1):
        if not (masks_dir / name).exists():
            raise FileNotFoundError(f"missing mask for {name}")
        shutil.copy(images_dir / name, root / "imagesTr" / f"DENTEX_{i:04d}_0000.png")
        shutil.copy(masks_dir / name, root / "labelsTr" / f"DENTEX_{i:04d}.png")
    (root / "case_mapping.json").write_text(json.dumps(
        {f"DENTEX_{i:04d}": n for i, n in enumerate(names, start=1)}, indent=1))

    if test_images_dir:
        for i, p in enumerate(sorted(Path(test_images_dir).glob("*.png")), start=1):
            shutil.copy(p, root / "imagesTs" / f"DENTEX_TEST_{i:04d}_0000.png")

    dataset_json = {
        "channel_names": {"0": "XRAY"},
        "labels": get_task(task).nnunet_labels(),
        "numTraining": len(names),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    (root / "dataset.json").write_text(json.dumps(dataset_json, indent=4))
    return root
