"""Predict masks, separate teeth, extract boxes and export COCO-style results.

Works on U-Net checkpoints from scripts/train_unet.py, or on label maps already
predicted by nnSAM (nnUNetv2_predict output folders) with --from-masks.

    # U-Net
    python scripts/predict.py --images data/test/xrays --sizes data/test/original_sizes.json \
        --quadrant runs/unet/quadrant/best.pt --enumeration runs/unet/enumeration/best.pt \
        --disease runs/unet/disease/best.pt --out results/unet_predictions.json

    # nnSAM predictions
    python scripts/predict.py --images data/test/xrays --sizes data/test/original_sizes.json --from-masks \
        --quadrant nnsam_pred/043 --enumeration nnsam_pred/044 --disease nnsam_pred/045 \
        --out results/nnsam_predictions.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dentex.data.dataset import SIZE, resize_pair
from dentex.export import to_records, write_predictions
from dentex.labels import get_task
from dentex.models.unet import UNet
from dentex.postprocess.boxes import boxes_from_labelmap

TASKS = ("quadrant", "enumeration", "disease")


def load_unet(path: str, task: str, device: str) -> UNet:
    state = torch.load(path, map_location="cpu")
    model = UNet(1, get_task(task).num_classes)
    model.load_state_dict(state["model"])
    return model.eval().to(device)


@torch.no_grad()
def predict_unet(model: UNet, image: Image.Image, device: str) -> tuple[np.ndarray, np.ndarray]:
    img, _ = resize_pair(image, None, SIZE)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)[None, None].to(device)
    probs = model(x).softmax(dim=1)[0].cpu().numpy()
    return probs.argmax(axis=0), probs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True)
    p.add_argument("--sizes", required=True, help="original_sizes.json written by make_masks.py")
    p.add_argument("--from-masks", action="store_true", help="task arguments are folders of predicted label maps")
    for t in TASKS:
        p.add_argument(f"--{t}", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    sizes = json.loads(Path(args.sizes).read_text())
    sources = {t: getattr(args, t) for t in TASKS if getattr(args, t)}
    models = {} if args.from_masks else {t: load_unet(src, t, args.device) for t, src in sources.items()}

    records = []
    for name, meta in tqdm(sorted(sizes.items()), desc="predict"):
        boxes = {}
        for task, src in sources.items():
            per_component = task == "disease"     # several lesions may share one diagnosis label
            if args.from_masks:
                labelmap = np.asarray(Image.open(Path(src) / name))
                boxes[task] = boxes_from_labelmap(labelmap, per_component=per_component)
            else:
                labelmap, probs = predict_unet(models[task], Image.open(Path(args.images) / name), args.device)
                boxes[task] = boxes_from_labelmap(labelmap, probs, per_component=per_component)
        records += to_records(meta["image_id"], (meta["width"], meta["height"]), **boxes, input_size=SIZE)

    write_predictions(records, args.out)
    print(f"{len(records)} boxes for {len(sizes)} images -> {args.out}")


if __name__ == "__main__":
    main()
