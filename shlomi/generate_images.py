#!/usr/bin/env python
"""Headless FLUX image generation for the shlomi/ pipeline.

Designed for long unattended runs (nohup / background). Resumable (skips images
already on disk) and reproducible (deterministic per-image seeds). Reads prompts
from shlomi/data/prompts.json via the local utils.py and writes undegraded images
to shlomi/data/synthetic_clean/{class}/ plus a manifest.csv.

Quick pilot (10 images per class = 50 total):
    nohup /home/benshise/my_new_project/.venv/bin/python shlomi/generate_images.py \
        --gpu 6 --per-class 10 > shlomi/gen.log 2>&1 &

Full run (all 300/class = 1500):  drop --per-class.

Watch:   tail -f shlomi/gen.log     (Ctrl+C stops watching, NOT the job)
Stop:    ps aux | grep generate_images   then   kill <pid>

Note: --size is the image RESOLUTION in pixels (default 512), NOT a count. 512 is
~4x faster than 1024 and step 03 degrades to ~128 anyway. FLUX.1-schnell is much
faster (no license, ~4 steps):  --model black-forest-labs/FLUX.1-schnell --steps 4 --guidance 0
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="6", help="CUDA device index to use")
ap.add_argument("--size", type=int, default=512,
                help="image RESOLUTION in pixels (not a count). 512 is ~4x faster than 1024")
ap.add_argument("--steps", type=int, default=20, help="num_inference_steps")
ap.add_argument("--guidance", type=float, default=3.5, help="guidance_scale (use 0 for FLUX.1-schnell)")
ap.add_argument("--images-per-prompt", type=int, default=1)
ap.add_argument("--per-class", type=int, default=0,
                help="cap prompts per class (0 = all 300). e.g. 10 for a quick 50-image pilot")
ap.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
args = ap.parse_args()

# Must be set BEFORE torch initialises CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                       # noqa: E402
from diffusers import FluxPipeline  # noqa: E402

# Import the local shlomi/utils.py (this file lives in shlomi/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils                       # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    prompts = utils.load_prompts()

    def plist(cls):
        """Prompts for a class, optionally capped by --per-class."""
        return prompts[cls] if args.per_class <= 0 else prompts[cls][:args.per_class]

    def seed_for(cls, p_i, k):
        """Deterministic per (class, prompt, image) so a pilot and a later full run agree."""
        return utils.SEED + utils.CLASS_TO_IDX[cls] * 1_000_000 + p_i * 100 + k

    total = sum(len(plist(c)) for c in utils.CLASS_NAMES) * args.images_per_prompt

    log(f"loading {args.model} on GPU {args.gpu} (fp16 + cpu offload) ...")
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    scope = f", {args.per_class}/class" if args.per_class > 0 else " (full set)"
    log(f"loaded. target = {total} images @ {args.size}px, {args.steps} steps{scope}.")

    def generate(prompt, seed):
        gen = torch.Generator("cpu").manual_seed(seed)
        return pipe(
            prompt,
            height=args.size, width=args.size,
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            max_sequence_length=512,
            generator=gen,
        ).images[0]

    log("generating ... (one progress line per image below)")
    t0 = time.time()
    made = done = 0
    for cls in utils.CLASS_NAMES:
        out_dir = utils.class_dir(utils.CLEAN_DIR, cls)
        for p_i, prompt in enumerate(plist(cls)):
            for k in range(args.images_per_prompt):
                done += 1
                fp = out_dir / f"{cls}_{p_i:04d}_{k}.jpg"
                if fp.exists():
                    continue
                generate(prompt, seed_for(cls, p_i, k)).save(fp, quality=95)
                made += 1
                rate = (time.time() - t0) / made
                eta_min = rate * (total - done) / 60
                log(f"{done}/{total}  [{cls}]  {rate:.0f}s/img  ETA ~{eta_min:.0f} min")

    # Manifest (rebuilt from the same deterministic loop; robust to resumes).
    utils.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(utils.MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "class", "seed", "model", "prompt"])
        for cls in utils.CLASS_NAMES:
            for p_i, prompt in enumerate(plist(cls)):
                for k in range(args.images_per_prompt):
                    fp = utils.CLEAN_DIR / cls / f"{cls}_{p_i:04d}_{k}.jpg"
                    if fp.exists():
                        w.writerow([str(fp.relative_to(utils.ROOT_DIR)), cls,
                                    seed_for(cls, p_i, k), args.model, prompt])
                        rows += 1

    log(f"DONE. {made} new images this run | {rows} total on disk | "
        f"elapsed {(time.time() - t0) / 60:.0f} min | manifest -> {utils.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
