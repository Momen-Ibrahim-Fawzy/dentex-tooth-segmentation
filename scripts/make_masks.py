"""Generate 512x512 training pairs (X-ray, label map) for one DENTEX task.

    python scripts/make_masks.py --json data/DENTEX/training_data/quadrant_enumeration/train_quadrant_enumeration.json \
        --images data/DENTEX/training_data/quadrant_enumeration/xrays --task enumeration \
        --out data/prepared/enumeration --refine border

--refine none     plain semantic masks
--refine overlap  teeth grown by --gap px and their intersections cut out (separates neighbours)
--refine border   overlap removal + a 3-class {background, tooth, border} map for instance separation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from dentex.data.coco import instance_labels, instance_masks, load_dentex, semantic_mask
from dentex.data.dataset import SIZE, resize_pair
from dentex.data.refine import add_border_class, separate_teeth


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--task", required=True, choices=["quadrant", "enumeration", "disease"])
    p.add_argument("--out", required=True)
    p.add_argument("--refine", default="none", choices=["none", "overlap", "border"])
    p.add_argument("--gap", type=int, default=6, help="pixels by which teeth are grown before cutting overlaps")
    p.add_argument("--border-size", type=int, default=5)
    args = p.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    sizes = {}

    for xray in tqdm(load_dentex(args.json), desc=f"masks [{args.task}/{args.refine}]"):
        semantic = semantic_mask(xray, args.task)
        if args.refine != "none":
            separated = separate_teeth(instance_masks(xray), instance_labels(xray, args.task), gap=args.gap)
            semantic = separated.astype(np.uint8)
            if args.refine == "border":
                semantic = add_border_class(separated, border_size=args.border_size)

        image = Image.open(Path(args.images) / xray.file_name)
        image, mask = resize_pair(image, Image.fromarray(semantic), SIZE)
        name = Path(xray.file_name).with_suffix(".png").name
        image.save(out / "images" / name)
        mask.save(out / "masks" / name)
        sizes[name] = {"image_id": xray.image_id, "width": xray.width, "height": xray.height}

    (out / "original_sizes.json").write_text(json.dumps(sizes, indent=1))
    print(f"wrote {len(sizes)} pairs to {out}")


if __name__ == "__main__":
    main()
