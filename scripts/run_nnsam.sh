#!/usr/bin/env bash
# Train and run nnSAM (frozen SAM encoder + nnU-Net) with 5-fold cross-validation for each DENTEX task.
#   bash scripts/run_nnsam.sh            # all tasks
#   bash scripts/run_nnsam.sh 043        # quadrant only
# Requires: pip install git+https://github.com/Kent0n-Li/nnSAM.git  (installs nnU-Net v2 with the nnSAM trainer)
set -euo pipefail
cd "$(dirname "$0")/.."

export MODEL_NAME=nnsam                                   # selects the nnSAM network in the nnSAM fork
export nnUNet_raw="${nnUNet_raw:-$PWD/nnsam/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PWD/nnsam/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PWD/nnsam/nnUNet_results}"

DATASETS=("${@:-043 044 045}")
for d in ${DATASETS[@]}; do
  echo "=== Dataset${d}_DENTEX ==="
  nnUNetv2_plan_and_preprocess -d "$d" -c 2d --verify_dataset_integrity
  for fold in 0 1 2 3 4; do
    # --c resumes from the latest checkpoint, so interrupted runs (e.g. Colab sessions) can continue
    if [ -f "$nnUNet_results/$MODEL_NAME/Dataset${d}_DENTEX/nnUNetTrainer__nnUNetPlans__2d/fold_${fold}/checkpoint_latest.pth" ]; then
      nnUNetv2_train "$d" 2d "$fold" --npz --c
    else
      nnUNetv2_train "$d" 2d "$fold" --npz
    fi
  done
  nnUNetv2_find_best_configuration "$d" -c 2d
  nnUNetv2_predict -i "$nnUNet_raw/Dataset${d}_DENTEX/imagesTs" -o "nnsam/predictions/${d}" -d "$d" -c 2d
done
