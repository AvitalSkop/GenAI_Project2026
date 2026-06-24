#!/usr/bin/env python
"""Make `finished_leftovers` by EDITING full-plate images with FLUX.1 Kontext.

Text-to-image can't render "absence of food" - naming a dish makes FLUX draw a full
serving. Instruction-based image editing solves it directly: take a full-plate image
and tell the model to *subtract* food (simulate someone having eaten most of it).

Pipeline:
    full-plate image  --(Kontext: "the person ate most of it, leave ~a quarter")-->  leftovers

This also keeps the leftovers image on the SAME plate/table/lighting as a full plate,
so the only thing that differs is the food amount (the actual label) - no style shortcut.

Source images: any folder of full-plate photos (default: shlomi/data/synthetic_clean/full).
Output: shlomi/data/synthetic_clean/finished_leftovers/ (overwrites the text2img ones).

Prereqs:
    - Accept the license for black-forest-labs/FLUX.1-Kontext-dev on HF (gated); cached token works.
    - diffusers with FluxKontextPipeline (>=0.35; you have 0.37.1).

Pilot (edit 5 full plates):
    python shlomi/edit_leftovers_kontext.py --gpu 0 --limit 5
Full (edit all images in --src):  drop --limit.
Headless:  nohup ... > shlomi/gen_kontext.log 2>&1 &
"""
import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0")
ap.add_argument("--src", default=None,
                help="folder of full-plate source images (default: synthetic_clean/full)")
ap.add_argument("--size", type=int, default=512, help="edit resolution (match the dataset)")
ap.add_argument("--steps", type=int, default=28, help="num_inference_steps")
ap.add_argument("--guidance", type=float, default=2.5, help="Kontext guidance_scale (~2.5-4)")
ap.add_argument("--limit", type=int, default=0, help="cap how many source images to edit (0 = all)")
ap.add_argument("--out", default=None,
                help="output folder (default: synthetic_clean/finished_leftovers, in place). "
                     "Use a new folder to keep the originals for an A/B compare.")
ap.add_argument("--overwrite", action="store_true", help="re-edit even if the output already exists")
ap.add_argument("--model", default="black-forest-labs/FLUX.1-Kontext-dev")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                            # noqa: E402
from diffusers import FluxKontextPipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                            # noqa: E402

SRC_DIR = Path(args.src) if args.src else (utils.CLEAN_DIR / "full")
OUT_DIR = Path(args.out) if args.out else (utils.CLEAN_DIR / "finished_leftovers")
MANIFEST = OUT_DIR / "kontext_manifest.csv"

# How much food the edit should leave (we want only ~a quarter / fifth).
EAT_AMOUNTS = [
    "only about a quarter of the food left",
    "roughly a fifth of the meal remaining",
    "just a few small scraps of food left",
    "only a small amount of food remaining in one corner",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_instruction(rng: random.Random) -> str:
    """A randomized 'eat-down' edit instruction (~1/2 cutlery, ~1/4 napkin)."""
    parts = [
        f"The person has eaten most of this meal. Edit the image so the same plate now has "
        f"{rng.choice(EAT_AMOUNTS)}:",
        "scattered messy scraps pushed to one side, smeared sauce, scattered crumbs and visible "
        "bite marks, while the rest of the plate is empty and bare.",
    ]
    parts.append("Leave a dirty fork or spoon resting in the plate."
                 if rng.random() < 0.5 else "No cutlery on the plate.")
    if rng.random() < 0.25:
        parts.append("Leave a crumpled, used paper napkin inside the plate.")
    parts.append("Keep the exact same plate, table surface, lighting and camera angle.")
    return " ".join(parts)


def main() -> None:
    if not SRC_DIR.is_dir() or not any(SRC_DIR.glob("*.jpg")):
        sys.exit(f"No source images in {SRC_DIR}. Generate some 'full' plates there first "
                 f"(e.g. generate_images.py --classes full).")

    srcs = sorted(SRC_DIR.glob("*.jpg"))
    if args.limit > 0:
        srcs = srcs[:args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading {args.model} on GPU {args.gpu} (fp16 + cpu offload) ...")
    pipe = FluxKontextPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    log(f"loaded. editing {len(srcs)} full plates -> leftovers @ {args.size}px, {args.steps} steps.")

    rows = []
    t0 = time.time()
    made = 0
    for i, src in enumerate(srcs):
        out_fp = OUT_DIR / f"finished_leftovers_{i:04d}_0.jpg"
        if out_fp.exists() and not args.overwrite:
            continue
        seed = utils.SEED + i
        rng = random.Random(seed)               # deterministic instruction per image
        instruction = build_instruction(rng)
        image = Image.open(src).convert("RGB").resize((args.size, args.size))
        edited = pipe(
            image=image,
            prompt=instruction,
            height=args.size, width=args.size,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]
        edited.save(out_fp, quality=95)
        made += 1
        rate = (time.time() - t0) / made
        log(f"{i + 1}/{len(srcs)}  {rate:.0f}s/img  ETA ~{rate * (len(srcs) - i - 1) / 60:.0f} min")
        rows.append([str(out_fp.relative_to(utils.ROOT_DIR)),
                     str(src.relative_to(utils.ROOT_DIR)), seed, instruction])

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "source_full_image", "seed", "instruction"])
        w.writerows(rows)
    log(f"DONE. {made} edited | elapsed {(time.time() - t0) / 60:.0f} min | manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
