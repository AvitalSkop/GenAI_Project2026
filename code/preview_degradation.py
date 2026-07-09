#!/usr/bin/env python
"""Preview the CCTV degradation (notebook 03) on a few images per class WITHOUT running the full
pipeline and WITHOUT saving anything to the dataset.

It reads the degradation code straight out of 03_degrade_and_augment.ipynb (the single source of
truth) - the `DEGRADE` knob dict and the GPU/torch `degrade_batch()` pipeline - degrades N images
per class on the fly, and writes a grid to code/degradation_preview.png so you can judge the
augmentation strength before the real run. Tune the ranges in the notebook's `DEGRADE` cell,
re-run this, repeat until it looks right.

Runs on CPU by default (the preview is tiny) so it never competes for a GPU; export
PREVIEW_DEVICE=cuda to force the GPU instead.

Usage:
    python preview_degradation.py            # 10 per class
    python preview_degradation.py 6          # 6 per class
    python preview_degradation.py 8 3        # 8 images/class, 3 degraded variants each
"""
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import utils  # noqa: E402

import matplotlib                  # noqa: E402
matplotlib.use("Agg")             # headless (no display): render straight to a file
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                       # noqa: E402
from PIL import Image             # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10        # images per class
VARIANTS = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # degraded versions shown per image
DEVICE = os.environ.get("PREVIEW_DEVICE", "cpu")         # CPU by default -> no GPU contention

# --- Pull DEGRADE + degrade_batch() straight out of the notebook (its single source of truth) ---
nb = json.loads((HERE / "03_degrade_and_augment.ipynb").read_text(encoding="utf-8"))
ns = {"utils": utils}
# The imports / constants the notebook's degradation cells rely on. These normally come from the
# notebook's config cell, which we deliberately skip so we don't inherit its GPU-selection side
# effects (it sets CUDA_VISIBLE_DEVICES). The pipeline itself is plain torch + PIL - no cv2.
exec(
    "import math, random\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn.functional as F\n"
    "from PIL import Image\n"
    "torch.set_grad_enabled(False)\n"   # inference-only preprocessing
    "SIZE = 224\n",                      # working resolution (matches the notebook)
    ns,
)
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "DEGRADE = {" in src or "def degrade_batch" in src:   # knobs cell + pipeline cell
        exec(src, ns)
try:
    degrade_batch = ns["degrade_batch"]
    load_to_tensor = ns["load_to_tensor"]
except KeyError as e:
    sys.exit(f"Could not find {e} in 03_degrade_and_augment.ipynb - has the notebook's "
             "degradation pipeline been renamed? This script expects DEGRADE + degrade_batch().")

INPUT_DIR = utils.CLEAN_DIR
classes = sorted(d.name for d in INPUT_DIR.iterdir() if d.is_dir()) if INPUT_DIR.is_dir() else []
if not classes:
    sys.exit(f"No class subfolders found under {INPUT_DIR} - nothing to preview.")
print("preview source :", INPUT_DIR)
print("classes        :", classes)
print("device         :", DEVICE)
print(f"showing        : {N} images/class, original + {VARIANTS} degraded variant(s) each")

# Fixed seeds so re-running the preview reflects only your DEGRADE edits, not RNG noise.
random.seed(utils.SEED); torch.manual_seed(utils.SEED); np.random.seed(utils.SEED)

exts = (".jpg", ".jpeg", ".png")
rows_per_class = 1 + VARIANTS                      # 1 original row + VARIANTS degraded rows
total_rows = rows_per_class * len(classes)
fig, axes = plt.subplots(total_rows, N, figsize=(2 * N, 2 * total_rows), squeeze=False)

for c, cls in enumerate(classes):
    files = sorted(p for p in (INPUT_DIR / cls).glob("*") if p.suffix.lower() in exts)[:N]
    base = c * rows_per_class
    for i in range(N):
        for r in range(rows_per_class):
            axes[base + r][i].axis("off")
        if i >= len(files):
            continue
        # original, resized to the working square so it lines up with the degraded rows below it
        axes[base][i].imshow(Image.open(files[i]).convert("RGB").resize((ns["SIZE"], ns["SIZE"])))
        # VARIANTS degraded copies of this image, each a fresh random draw, in one batched call
        batch = torch.stack([load_to_tensor(files[i]) for _ in range(VARIANTS)]).to(DEVICE)
        deg = degrade_batch(batch).cpu()
        for v in range(VARIANTS):
            arr = (deg[v].clamp(0, 1) * 255).byte().numpy().transpose(1, 2, 0)
            axes[base + 1 + v][i].imshow(arr)
    axes[base][0].set_title(f"{cls}  -  original (top) / degraded (below)", loc="left", fontsize=11)

fig.suptitle("Degradation preview  -  on the fly, nothing saved to the dataset", fontsize=13)
plt.tight_layout()
out = HERE / "degradation_preview.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved ->", out)
