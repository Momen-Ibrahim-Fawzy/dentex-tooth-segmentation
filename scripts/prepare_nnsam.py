"""Create the nnU-Net v2 raw datasets used to train nnSAM (Dataset043/044/045_DENTEX).

    python scripts/prepare_nnsam.py --prepared data/prepared --nnunet-raw nnsam/nnUNet_raw
expects data/prepared/<task>/{images,masks} from scripts/make_masks.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dentex.nnsam import build_raw_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared", required=True)
    p.add_argument("--nnunet-raw", required=True)
    p.add_argument("--test-images", default=None)
    p.add_argument("--tasks", nargs="+", default=["quadrant", "enumeration", "disease"])
    args = p.parse_args()
    for task in args.tasks:
        root = Path(args.prepared) / task
        out = build_raw_dataset(task, root / "images", root / "masks", args.nnunet_raw, args.test_images)
        print(f"{task:12s} -> {out}")


if __name__ == "__main__":
    main()
