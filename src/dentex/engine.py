"""U-Net training and validation loops."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dentex.losses import DiceCELoss
from dentex.metrics import SegmentationMeter, binary_scores


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, num_classes: int, device: str) -> dict[str, float]:
    """Macro per-class scores plus pooled foreground-vs-background scores ("fg_*")."""
    model.eval()
    meter = SegmentationMeter(num_classes)
    fg = SegmentationMeter(2)
    for batch in loader:
        pred = model(batch["image"].to(device)).argmax(dim=1).cpu().numpy()
        target = batch["mask"].numpy()
        meter.update(pred, target)
        fg.update((pred > 0).astype(np.int64), (target > 0).astype(np.int64))
    result = meter.summary()
    (tn, fp), (fn, tp) = fg.cm.astype(float)
    result.update({f"fg_{k}": v for k, v in binary_scores(tp, fp, fn, tn).items()})
    return result


def train(model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader, *, num_classes: int,
          epochs: int, lr: float, out_dir: str | Path, device: str = "cuda", patience: int = 15) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion = DiceCELoss(num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history, best, stale = [], -np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for batch in train_loader:
            logits = model(batch["image"].to(device))
            losses = criterion(logits, batch["mask"].to(device))
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            running += losses["loss"].item()

        val = validate(model, val_loader, num_classes, device)
        scheduler.step(val["dice"])
        row = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), **val,
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        print(f"epoch {epoch:3d} loss={row['train_loss']:.4f} dice={val['dice']:.4f} "
              f"fg_iou={val['fg_iou']:.4f} ({row['seconds']:.0f}s)")

        if val["dice"] > best:
            best, stale = val["dice"], 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch}")
                break
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {"best_dice": best, "epochs": len(history)}
