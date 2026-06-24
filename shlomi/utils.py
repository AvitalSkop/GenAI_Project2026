"""
utils.py - shared, single-source-of-truth constants and helpers for the
Plate Status Detection project.

Everything that must stay identical across every notebook lives here: the
random seed, the canonical class list and ordering, the binary decision rule,
project paths, and the attribute pools + prompt builder used by
01_generate_prompts.

Import this module from every notebook; never redefine these values ad hoc.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42  # fixed seed used everywhere so runs are reproducible

# ---------------------------------------------------------------------------
# Class taxonomy. The first four are ordered by how much is on the plate
# (low -> high); `unclassified` is a special trailing class, off that axis.
# ---------------------------------------------------------------------------
# The order is fixed so label indices and confusion-matrix axes are stable
# across notebooks. The index of each class is its position in this list.
CLASS_NAMES = [
    "clean",               # 0 - pristine, unused plate (no food/crumbs)
    "empty",               # 1 - used plate eaten bare (crumbs/residue)
    "finished_leftovers",  # 2 - small leftovers or garbage (napkin/paper)
    "full",                # 3 - a moderate-to-full serving (merges old semi_full + full)
    "unclassified",        # 4 - too degraded to identify (borrows prompts; corrupted in step 03)
]

# Classes that have their own prompts. `unclassified` is excluded: it borrows
# random prompts from these in build_prompts(), and step 03 corrupts those images
# until the plate state is unidentifiable.
CONTENT_CLASSES = ["clean", "empty", "finished_leftovers", "full"]
UNCLASSIFIED = "unclassified"
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# Binary decision rule (NON-MONOTONIC by design: a `clean` plate has the least
# on it yet maps to "do not clear" because it is a freshly set plate).
# `unclassified` is a third outcome - the image is too degraded to decide, so we
# abstain ('uncertain') rather than guess; the safe deployment fallback is to
# NOT auto-clear a plate we cannot assess.
CLEAR_CLASSES = {"empty", "finished_leftovers"}
DONT_CLEAR_CLASSES = {"clean", "full"}
UNCERTAIN_CLASSES = {"unclassified"}


def to_binary(class_name: str) -> str:
    """Map a class to an action: 'clear', 'do_not_clear', or 'uncertain'."""
    if class_name in CLEAR_CLASSES:
        return "clear"
    if class_name in DONT_CLEAR_CLASSES:
        return "do_not_clear"
    if class_name in UNCERTAIN_CLASSES:
        return "uncertain"
    raise ValueError(f"Unknown class: {class_name!r}")


# ---------------------------------------------------------------------------
# Project paths (resolved relative to the repo root so they work locally and
# on Colab, as long as this file stays in code/).
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent              # the shlomi/ workspace root (utils.py lives here)
DATA_DIR = ROOT_DIR / "data"
PROMPTS_PATH = DATA_DIR / "prompts.json"
CLEAN_DIR = DATA_DIR / "synthetic_clean"                # undegraded images, per-class subfolders
DEGRADED_DIR = DATA_DIR / "synthetic_degraded"          # degraded images, per-class subfolders
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT_DIR / "results"
PILOT_DIR = DATA_DIR / "_pilot"                         # small QA sample written by the step 02 pilot
MANIFEST_PATH = CLEAN_DIR / "manifest.csv"              # filepath/class/seed/model/prompt of generated images

# NOTE: the shared train/val/test split function will be added here in step 04
# so that every notebook imports the exact same split. Do not re-split ad hoc.


def class_dir(base: Path, class_name: str) -> Path:
    """Return <base>/<class_name>, creating it if needed.

    Handy for the per-class subfolders of synthetic_clean / synthetic_degraded /
    the step-02 pilot. Pure-stdlib so importing utils stays dependency-light.
    """
    d = Path(base) / class_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Prompt generation (used by 01_generate_prompts)
# ---------------------------------------------------------------------------
# Nuisance / context attributes - varied across ALL classes so the model keys
# on food amount, not on these incidental cues.
#
# Each pool is a list of (value, weight) pairs; weights are relative. They make
# the COMMON real case dominant (white-ish, round, top-down) while keeping a
# diverse minority, so the model performs on typical restaurant plates yet stays
# robust to rarer ones ("dominant-but-diverse"). Weights are easy to tune - e.g.
# to match the real_restaurant_cctv reference photos.
#
# KEY INVARIANT: these pools are sampled INDEPENDENTLY of the class, so plate
# appearance carries no label information and cannot become a shortcut.

# Plate style bundles color + material/texture coherently (article baked in).
# White-ish styles stay the plurality (realistic), with a wide diverse tail.
PLATE_STYLES = [
    ("a glossy white porcelain", 26),
    ("an off-white ceramic", 12),
    ("a matte cream-colored ceramic", 9),
    ("a plain white stoneware", 9),
    ("a blue-rimmed white porcelain", 7),
    ("a beige stoneware", 7),
    ("a pale grey ceramic", 6),
    ("a dark slate", 5),
    ("a rustic terracotta", 4),
    ("a speckled grey stoneware", 4),
    ("a black matte", 4),
    ("a dark green ceramic", 3),
    ("a navy blue ceramic", 3),
    ("a hand-painted patterned ceramic", 3),
    ("a hammered metal", 2),
    ("a bamboo", 2),
]
PLATE_SHAPES = [
    ("round dinner plate", 48),
    ("oval plate", 11),
    ("square plate", 11),
    # NOTE: never name a FOOD here. "pasta bowl" / "deep dish (pizza)" are literal
    # nouns FLUX renders as food, contaminating empty/clean plates. Keep shapes
    # food-free so the shape pool carries no label signal.
    ("shallow wide bowl", 10),
    ("deep round bowl", 9),
    ("rectangular plate", 6),
    ("small round appetizer plate", 5),
]
TABLE_SURFACES = [
    ("a dark wooden table", 24),
    ("a white tablecloth", 22),
    ("a light wooden table", 14),
    ("a rustic wooden table", 10),
    ("a marble table", 10),
    ("a checkered tablecloth", 8),
    ("a slate surface", 5),
    ("a woven placemat", 4),
    ("a glass table", 3),
]
# A ceiling camera almost always looks straight down, so top-down dominates.
CAMERA_ANGLES = [
    ("directly top-down", 70),
    ("a near top-down angle", 18),
    ("a slight overhead angle", 12),
]
CUTLERY_STATES = [
    ("a fork and knife resting on it", 22),
    ("a fork and knife laid beside it", 20),
    ("no cutlery visible", 22),
    ("a spoon resting on it", 12),
    ("chopsticks laid beside it", 12),
    ("a folded napkin beside it", 12),
]
LIGHTINGS = [
    ("dim restaurant lighting", 26),
    ("warm tungsten light", 22),
    ("cool fluorescent overhead light", 18),
    ("soft natural daylight", 16),
    ("bright overhead lighting", 10),
    ("moody low light", 8),
]

# Food items used to diversify the food-bearing classes so images do not collapse
# onto a single dish. Bare noun phrases (no leading article) so they read well in
# every phrasing, e.g. "a plate of {food}" / "a few bites of {food}". Many cuisines
# and dish types for maximum visual variety.
FOOD_ITEMS = [
    "pasta with tomato sauce", "spaghetti bolognese", "creamy fettuccine", "lasagna",
    "pizza", "mushroom risotto", "mac and cheese",
    "fried rice", "noodles with vegetables", "ramen", "pad thai", "sushi",
    "dumplings", "dim sum", "stir-fried vegetables and rice",
    "chicken curry with rice", "butter chicken with naan", "lentil dahl",
    "burger and fries", "fish and chips", "grilled steak and mashed potatoes",
    "roast chicken and vegetables", "grilled salmon and asparagus", "shrimp and rice",
    "Greek salad", "caesar salad", "mixed green salad", "fruit salad",
    "vegetable soup", "tomato soup",
    "hummus and pita", "falafel and salad", "shawarma and rice", "beef kebab and rice",
    "tacos", "beef burrito", "nachos",
    "pancakes with syrup", "scrambled eggs and toast", "omelette and salad",
]

# Per-class content phrasings - THIS is what actually defines the label.
# "{food}" is filled from FOOD_ITEMS for the food-bearing classes.
CLASS_CONTENTS = {
    "clean": [
        "a pristine, completely unused clean plate with no food and no crumbs at all",
        "a spotless freshly-set plate, nothing on it, no residue whatsoever",
        "a brand-new looking spotless clean plate, ready to be served on",
        "an immaculate empty plate straight from the kitchen, perfectly clean",
        "a clean, polished plate with a spotless surface and no marks or food",
    ],
    "empty": [
        "an empty plate that has been eaten from, bare of food but with light crumbs and sauce smears",
        "a used plate scraped clean of food, only leftover crumbs and a few sauce streaks remaining",
        "a finished empty plate with smudges, crumbs and dried sauce residue, no real food left",
        "an empty plate after a meal, just grease marks and a few crumbs, no food",
        "a wiped-out plate with only faint sauce stains and scattered crumbs, no food remaining",
    ],
    # Lead with the EMPTINESS and use FLUX's "with emphasis on" phrasing (it has no
    # prompt weights). Describe the scattered REMAINS, not the full dish, or FLUX
    # renders a full serving of {food} and ignores "a few bites". Strong remnant
    # words (scraps, smears, crumbs, half-eaten, mostly gone) anchor the small amount.
    "finished_leftovers": [
    "a mostly eaten plate, with emphasis on how little is left: just a few small scraps of {food} pushed to one side, with sauce smears and crumbs across an otherwise empty plate",
    "the messy remains of a meal, only a couple of leftover bites of {food} among streaks of sauce and scattered crumbs, with most of the food already eaten",
    "a nearly empty plate after eating, just a small leftover scrap of {food} in one corner, the rest wiped down to sauce stains and crumbs",
    "a half-eaten plate with only a little {food} remaining in one corner, with emphasis on the empty messy surface, leftover sauce and crumbs",
    ],
    # `full` spans moderate -> full (the old semi_full and full classes merged).
    # Add "casually/unevenly served" so the food looks real and homestyle, not the
    # over-symmetrical studio plating FLUX defaults to.
    "full": [
    "a full plate of {food}, a complete fresh portion, casually served and a bit uneven",
    "a generous serving of {food} piled unevenly across the whole plate, plated homestyle and slightly messy",
    "a plate heaped with a large helping of {food}, sauce splashed naturally, not neatly arranged",
    "a plate filled with a hearty portion of {food}, casually served so it looks real and unstaged",
    ],
}

# NOTE: deliberately NO "security camera" / "CCTV" / "surveillance" wording.
# Those phrases make the diffusion model burn a fake HUD overlay (timestamp,
# "REC", camera id) into the image, which the classifier could then cheat on.
# We keep the diffusion output a CLEAN photo and add the realistic CCTV
# degradation (low-res, noise, blur, JPEG) separately and controllably in
# step 03 - that separation is also what keeps the degradation ablation valid.
PROMPT_TEMPLATE = (
    "{style} {shape} on {surface}, viewed from {angle}, "
    "with {contents}, {cutlery}, under {lighting}. "
    "A candid, slightly imperfect overhead phone snapshot of a single plate "
    "filling the frame, natural and unstaged."
)


def _weighted(rng: random.Random, pairs: list[tuple[str, int]]) -> str:
    """Pick one value from a list of (value, weight) pairs (weights are relative)."""
    values = [v for v, _ in pairs]
    weights = [w for _, w in pairs]
    return rng.choices(values, weights=weights, k=1)[0]


def _make_contents(rng: random.Random, class_name: str) -> str:
    """Pick a content phrasing for a class, filling in a random food where needed."""
    phrasing = rng.choice(CLASS_CONTENTS[class_name])
    if "{food}" in phrasing:
        phrasing = phrasing.format(food=rng.choice(FOOD_ITEMS))
    return phrasing


def build_prompts(n_per_class: int = 300, seed: int = SEED) -> dict[str, list[str]]:
    """
    Build attribute-based text-to-image prompts for every class.

    For each class we randomly sample unique combinations of the nuisance
    attributes (plate, table, angle, cutlery, lighting) plus a class-specific
    content phrase. Nuisance attributes are varied across every class so the
    classifier cannot cheat on them.

    Returns a dict {class_name: [prompt, ...]} with n_per_class prompts each.
    Deterministic for a fixed seed.
    """
    rng = random.Random(seed)
    prompts: dict[str, list[str]] = {}

    # 1) Content classes - each gets its own unique, attribute-based prompts.
    for class_name in CONTENT_CLASSES:
        seen: set[str] = set()
        out: list[str] = []
        attempts = 0
        # Cap attempts so we never loop forever if a pool is unexpectedly small.
        while len(out) < n_per_class and attempts < n_per_class * 50:
            attempts += 1
            prompt = PROMPT_TEMPLATE.format(
                style=_weighted(rng, PLATE_STYLES),
                shape=_weighted(rng, PLATE_SHAPES),
                surface=_weighted(rng, TABLE_SURFACES),
                angle=_weighted(rng, CAMERA_ANGLES),
                contents=_make_contents(rng, class_name),
                cutlery=_weighted(rng, CUTLERY_STATES),
                lighting=_weighted(rng, LIGHTINGS),
            )
            prompt = prompt[0].upper() + prompt[1:]  # capitalize the sentence start
            if prompt not in seen:
                seen.add(prompt)
                out.append(prompt)
        prompts[class_name] = out

    # 2) `unclassified` borrows random prompts from the content classes. The
    #    images these produce are NORMAL plates; step 03 then corrupts them so
    #    heavily that the plate state can no longer be identified. This class is
    #    therefore only meaningful once degradation is applied (see step 03).
    pool = [p for class_name in CONTENT_CLASSES for p in prompts[class_name]]
    prompts[UNCLASSIFIED] = rng.sample(pool, n_per_class)

    return prompts


def save_prompts(prompts: dict[str, list[str]], path: Path = PROMPTS_PATH) -> None:
    """Write the prompts dict to JSON (keyed by class), creating data/ if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def load_prompts(path: Path = PROMPTS_PATH) -> dict[str, list[str]]:
    """Load the prompts dict from JSON (keyed by class)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
