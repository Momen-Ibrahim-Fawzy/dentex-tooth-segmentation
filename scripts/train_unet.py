"""Train a U-Net for one DENTEX task (80/20 train/validation split).

    python scripts/train_unet.py --config configs/unet.yaml --task quadrant
    python scripts/train_unet.py --config configs/unet.yaml --task border   # {bg, tooth, border} for instances
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dentex.data.dataset import SegmentationDataset, split_names
from dentex.engine import train
from dentex.labels import get_task
from dentex.models.unet import UNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True, choices=["quadrant", "enumeration", "disease", "border"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])

    root = Path(cfg["data_root"].format(task=args.task))
    if args.task == "border":
        label_task, num_classes = None, 3
    else:
        label_task, num_classes = args.task, get_task(args.task).num_classes

    names = sorted(p.name for p in (root / "images").glob("*.png"))
    train_names, val_names = split_names(names, cfg["val_fraction"], cfg["seed"])
    train_loader = DataLoader(SegmentationDataset(root, label_task, train_names, augment=True),
                              batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["workers"], drop_last=True)
    val_loader = DataLoader(SegmentationDataset(root, label_task, val_names),
                            batch_size=cfg["batch_size"], num_workers=cfg["workers"])

    model = UNet(1, num_classes, base=cfg["base_channels"], dropout=cfg["dropout"])
    summary = train(model, train_loader, val_loader, num_classes=num_classes, epochs=cfg["epochs"],
                    lr=cfg["lr"], out_dir=Path(cfg["output_dir"]) / args.task, device=args.device,
                    patience=cfg["patience"])
    print(summary)


if __name__ == "__main__":
    main()
