# CLAUDE.md — Plate Status Detection

GenAI course final project. A classifier that decides whether a restaurant table's plate should be
**cleared**, from a single low-quality security-camera image. Training/eval data is **100% synthetic**
(FLUX.1-dev generation + CCTV-style degradation).

## Layout
```
Project Main Folder/
  code/     # notebooks 01-06 + all .py + run_in_container.sh + secrets.env + requirements.txt
  data/     # prompts.json + image sets (Diff_DataSet, synthetic_degraded, splits, real_test)
  weights/  # trained model weights (.pth)                      [gitignored]
```
`code/utils.py` is the **single source of truth** (seed, class list, binary rule, paths, prompt
builder). Every notebook imports it — never redefine those values ad hoc. `data/` and `weights/` are
**siblings** of `code/` (utils resolves them via `PROJECT_DIR = ROOT_DIR.parent`). The data root is
overridable with the `PLATE_DATA_ROOT` env var **without editing any cell** (used on the GPU box).

## Critical domain rules (easy to get wrong — get these right)
- **Input is ONE still image of ONE plate** (not video, not a sequence). The crop is done upstream and
  is out of scope — assume a tight, single-plate crop.
- **3 classes, defined by the plate's visible state** (never by cutlery, never by whether a diner is present):
  - `clean` — pristine, **unused** plate; no food, no crumbs → **do not clear**
  - `finished` — the meal is **over**: eaten bare (crumbs/sauce/residue) **or** only small leftovers → **clear**
  - `full` — a **moderate-to-full** serving of food → **do not clear**
- **Binary rule (NON-MONOTONIC by design):** `clear = {finished}`, `do not clear = {clean, full}`.
  A `clean` plate has the *least* on it yet is *do not clear* (freshly set), `finished` (scraps) is
  *clear*, and `full` (the most) is *do not clear* — so "clear" is a band in the **middle** of the
  food-amount axis, not a threshold. This non-monotonic boundary is why a fine-grained model is needed.
- **`clean` vs `finished` is the subtlest pair — watch it.** Pristine vs eaten-bare-with-a-few-crumbs
  blurs under degradation and dominates the real-image errors. Treat it as error-analysis material, not a bug.
- **Cutlery / napkins are nuisance attributes**, varied randomly with **identical distributions across
  classes**, so the model keys on food amount and these can never become a shortcut.
- **`data/real_test/` images are a bonus sanity check / calibration ONLY — never training or val data.**
  The training/eval set is 100% synthetic.

## Pipeline (in `code/`, run in order)
| Step | File | Reads → Writes |
|---|---|---|
| 01 | `01_generate_prompts.ipynb` | — → `data/prompts.json` (attribute-based; identical nuisance distributions across classes) |
| 02 | `02_generate_images.ipynb` / `generate_images.py` | prompts → `data/Diff_DataSet_vN/{class}/` (FLUX.1-dev; versioned) |
| 03 | `03_degrade_and_augment.ipynb` | `data/Diff_DataSet/` → `data/synthetic_degraded/{class}/` (CCTV degradation — **novelty #1**) |
| 04 | `04_split_dataset.ipynb` | Diff_DataSet + degraded → `data/splits/{train,val,test}/{class}/` |
| 05 | `05_train_model.ipynb` | `data/splits/` → `weights/` (ResNet18 linear-probe → partial fine-tune; + CLIP baseline) |
| 06 | `06_train_model_Resnet50.ipynb` | `data/splits/` → `weights/resnet50_best.pth` (ResNet50, alternate backbone) |

- **`Diff_DataSet/`** is the curated **undegraded** set that step 03 reads (`utils.CLEAN_DIR`). NEW
  generation runs go to fresh versioned folders `Diff_DataSet_v1/_v2/...` (`utils.next_versioned_dir()`)
  so the curated set is never overwritten.
- **03 is class-agnostic:** `degrade_image()` never sees the label, so no degradation can correlate with
  the class. Downscale (always) + a *random subset* of blur / noise / lighting / colour-cast / vignette /
  JPEG / perspective, 2–5 copies per image, deterministic seeds.
- **05:** ResNet18 **linear probe** (frozen backbone + new head) → optional **partial fine-tune**
  (unfreeze `layer4` + head only — full-backbone FT overfits the small synthetic set). Reports accuracy,
  macro-F1, confusion matrix, the **binary clear/do-not-clear** accuracy, a **CLIP zero-shot baseline**,
  and a **real-test per-image CSV** (`weights/real_test_predictions.csv`).

## Hard technical gotchas
- **Never put "security camera" / "CCTV" / "surveillance" in a FLUX prompt** — it burns a fake HUD
  (timestamp, "REC", camera id) into the image, a shortcut the classifier would learn instead of food
  amount. Keep the diffusion output CLEAN; add the CCTV degradation only in step 03 (this also keeps
  the with/without-degradation ablation valid).
- **GPU-heavy work runs on the container**, not locally (FLUX generation, training).
- **FLUX.1-dev is gated** — needs a free HF account + accepted license + a read token (`HF_TOKEN`). Never commit it.
- **CLIP loads with `use_safetensors=True`** — newer `transformers` refuses `torch.load` of `.bin`
  weights on torch < 2.6 (CVE-2025-32434); safetensors sidesteps it without upgrading torch.
- **Eval `Resize((224,224))` squashes non-square images** — real oval/landscape plates get distorted;
  use `Resize(224) + CenterCrop(224)` if that matters.

## Conventions
- **Fixed seeds everywhere** (`SEED=42`) — runs reproducible; the degraded set and the splits are deterministic.
- **Keep both undegraded and degraded copies on disk** (`Diff_DataSet/` + `synthetic_degraded/`) so the
  with/without-degradation ablation can run.
- **`code/secrets.env`** (gitignored): `WANDB_API_KEY` (shared team W&B) and `HF_TOKEN`. `05` logs to
  W&B **only if `WANDB_API_KEY` is set**, otherwise it runs untracked. Never commit secrets.
- **gitignore:** all image data (`data/`, `*.jpg`/`*.png`) and weights (`*.pth`) are ignored; only
  `prompts.json` (force-added) and the code are tracked.
- Notebooks must run top-to-bottom; `requirements.txt` stays pinned and complete.
- **Working in a notebook on the container:** pull *before* you edit (pulling after editing causes
  conflicts). Order: pull → edit → commit.

## How to run (GPU container)
```bash
cd code
python -m venv ~/plate-train && source ~/plate-train/bin/activate
pip install -r requirements.txt          # if a CUDA torch already works, drop the torch/torchvision pins
# 1) put data at  data/Diff_DataSet/{clean,finished,full}/
# 2) put your HF + W&B keys in  code/secrets.env
nvidia-smi                                # pick a free GPU
bash run_in_container.sh 0                # runs 03 -> 04 -> 05 headless on GPU 0
```
Preview before committing to a full run: `python preview_degradation.py 10` (step-03 look) and
`python preview_train_transform.py` (step-05 live augmentation) — each writes a PNG.

## Current state
The full 3-class pipeline runs end to end. ResNet18 reaches ~0.95 synthetic val accuracy and **~85% on
the real-image sanity set** (errors concentrate on the `clean`↔`finished` boundary, as expected).
ResNet50 (`06`) is the alternate backbone. W&B project: `plate-classification`.
