#!/usr/bin/env bash
# One-shot pipeline runner for the university GPU container.
#
#   bash "<Project Main Folder>/code/run_in_container.sh" [GPU_INDEX]
#
# Sets the environment (GPU, secrets, data location) and executes the notebooks headless, in order,
# with NO per-run cell editing. It locates itself, so you can call it from anywhere.
set -euo pipefail
cd "$(dirname "$0")"                          # -> code/ (this script's own folder)

# 1) Pick a GPU. On a shared box, choose a free one (check `nvidia-smi`). Default 0.
export CUDA_VISIBLE_DEVICES="${1:-0}"

# 2) Load secrets (W&B key + HF token). Never committed - secrets.env sits next to this script (code/).
set -a
[ -f secrets.env ] && . secrets.env
set +a

# 3) Data root = the sibling  data/  folder (holds Diff_DataSet/{clean,finished,full}/, splits/, ...).
#    Override by exporting PLATE_DATA_ROOT before calling.
export PLATE_DATA_ROOT="${PLATE_DATA_ROOT:-$(cd .. && pwd)/data}"

echo "GPU=$CUDA_VISIBLE_DEVICES | DATA=$PLATE_DATA_ROOT | W&B=${WANDB_API_KEY:+on}${WANDB_API_KEY:-off}"

# 4) Execute the training pipeline in order (03 degrade -> 04 split -> 05 train). Already in code/.
#    (Step 02 / FLUX generation is run separately via generate_images.py.)
for nb in 03_degrade_and_augment 04_split_dataset 05_train_model; do
    echo "=== running ${nb}.ipynb ==="
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 "${nb}.ipynb"
done
echo "PIPELINE DONE. Trained weights + real-test CSV are under ../weights/ ; metrics on W&B."
