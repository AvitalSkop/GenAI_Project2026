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
# Real restaurant cameras are usually wall/corner-mounted, so we want MOSTLY angled
# shots: ~30% straight top-down, ~70% at a clear oblique angle. Weak words ("slight
# tilt") + "high overhead angle" made FLUX default to top-down, so we now use STRONG,
# CONCRETE perspective cues (camera to the side, plate seen in perspective, round rim
# as an ellipse) and state the sharpness explicitly (mild / clear three-quarter /
# sharp steep). No degree numbers (avoids a stray number rendering into the image).
CAMERA_ANGLES = [
    ("directly top-down, the camera straight above the plate looking down", 32),
    ("a mild oblique angle, the camera a little to the side so the plate is seen slightly in perspective rather than flat overhead", 22),
    ("a clear three-quarter angle, the camera well off to one side looking down and across the plate, the round rim appearing as an ellipse in perspective", 24),
    ("a sharp, steep oblique angle from above and to the side, the plate seen at a strong slant in clear perspective", 22),
]
CUTLERY_STATES = [
    ("a fork and knife resting on it", 25),
    ("a fork and knife laid beside it", 23),
    ("no cutlery visible", 25),
    ("a spoon resting on it", 14),
    ("chopsticks laid beside it", 13),
    # NOTE: no "napkin" option here - napkins are added ONLY to eaten-from plates
    # (empty / finished_leftovers) via USED_NAPKINS below, never to clean/full.
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
    # `empty` = a USED plate eaten bare: only crumbs / sauce smears / residue, no
    # real food. Diversified with the old near-empty "finished_leftovers" wording
    # (those described essentially-empty plates, so they belong here, not in
    # finished_leftovers which must now show actual leftover food).
    "empty": [
        "an empty plate that has been eaten from, bare of food but with light crumbs and sauce smears",
        "a used plate scraped clean of food, only leftover crumbs and a few sauce streaks remaining",
        "a finished empty plate with smudges, crumbs and dried sauce residue, no real food left",
        "an empty plate after a meal, just grease marks and a few crumbs, no food",
        "a wiped-out plate with only faint sauce stains and scattered crumbs, no food remaining",
        "a mostly empty plate after eating, just a smear of sauce and a few scattered crumbs, essentially no food",
        "a bare plate at the end of a meal, only dried sauce streaks and small crumbs remaining",
        "an almost spotless used plate with a faint food smear and a couple of crumbs, the food all eaten",
        "a cleaned-off plate showing only grease marks, sauce stains and a stray crumb or two",
    ],
    # GOAL: "most of the dish was eaten" - an almost-bare plate, ready to clear. Make
    # the EMPTY PLATE the subject ("almost entirely empty", "scraped nearly clean",
    # "most of the food already eaten") and reduce {food} to a tiny smear/remnant, so
    # it reads clearly emptier than `full` (the boundary).
    "finished_leftovers": [
        "an almost entirely empty plate at the end of a meal, most of the food already eaten, with only a tiny smear of {food}, a few crumbs and sauce streaks left in one corner",
        "a plate scraped nearly clean, just a single small leftover bite of {food} remaining among smudges of dried sauce, with the rest of the meal already eaten",
        "a mostly bare plate after eating, only a small remnant of {food} and scattered crumbs left, clearly almost finished and ready to be cleared",
        "a largely empty plate, the meal nearly done and most of it eaten, with just a little leftover {food} pushed to one side and sauce smears across the plate",
    ],
    # `full` spans MODERATE -> full (old semi_full + full merged) and is "do not
    # clear". Mix complete portions with meals-in-progress that still have PLENTY of
    # food left (so they stay clearly fuller than finished_leftovers).
    "full": [
        "a full plate of {food}, a complete fresh portion, casually served and a bit uneven",
        "a generous serving of {food} piled unevenly across the whole plate, plated homestyle and slightly messy",
        "a plate heaped with a large helping of {food}, sauce splashed naturally, not neatly arranged",
        "a half-eaten plate of {food} with a substantial amount still left, the meal clearly still in progress and far from finished",
        "a plate of {food} partway through the meal, some already eaten but plenty of food still remaining",
        "a moderately full plate of {food}, a normal portion with just a couple of bites taken, most of the meal still there",
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
    "A casual, slightly imperfect amateur phone snapshot of a single plate filling "
    "the frame, with natural uneven lighting, faint grain and slightly soft focus - "
    "a real candid photo taken in a restaurant, not a polished studio food shot."
)

# Used paper napkins / napkins left on the plate - a strong "finished eating" cue.
# Appended to ~20% of `empty` and `finished_leftovers` prompts (plates someone has
# eaten from); NEVER on `clean` (pristine) or `full` (still being eaten).
NAPKIN_CLASSES = {"empty", "finished_leftovers"}
NAPKIN_PROB = 0.20
USED_NAPKINS = [
    "a crumpled used paper napkin left on the plate",
    "a scrunched-up used napkin discarded on the plate",
    "a balled-up dirty paper napkin sitting on the plate",
    "a used napkin tossed onto the plate",
]


def _weighted(rng: random.Random, pairs: list[tuple[str, int]]) -> str:
    """Pick one value from a list of (value, weight) pairs (weights are relative)."""
    values = [v for v, _ in pairs]
    weights = [w for _, w in pairs]
    return rng.choices(values, weights=weights, k=1)[0]


def _make_contents(rng: random.Random, class_name: str) -> str:
    """Pick a content phrasing for a class, filling in food and (for eaten-from plates) a napkin."""
    phrasing = rng.choice(CLASS_CONTENTS[class_name])
    if "{food}" in phrasing:
        phrasing = phrasing.format(food=rng.choice(FOOD_ITEMS))
    # ~1/3 of empty / finished_leftovers plates get a used napkin (a "done eating" cue).
    if class_name in NAPKIN_CLASSES and rng.random() < NAPKIN_PROB:
        phrasing = f"{phrasing}, with {rng.choice(USED_NAPKINS)}"
    return phrasing


def build_prompts(n_per_class: int = 300, seed: int = SEED) -> dict[str, list[str]]:
    """
    Build attribute-based text-to-image prompts for every class.

    The nuisance attributes (plate style/shape, table surface, camera angle, cutlery,
    lighting) are sampled ONCE into a shared set of unique combinations and reused for
    the SAME index across every class. So all classes have an IDENTICAL distribution of
    these attributes (no per-class difference at all) - they carry zero label info and
    cannot become a shortcut. Only the per-class CONTENT (food state, plus a used napkin
    on ~20% of the eaten-from plates) differs by class.

    Returns {class_name: [prompt, ...]} with n_per_class unique prompts each.
    Deterministic for a fixed seed.
    """
    rng = random.Random(seed)

    # 1) Shared, UNIQUE nuisance combinations - the same list for every class, so the
    #    plate/surface/angle/cutlery/lighting distributions are byte-identical across
    #    classes (uniqueness barely perturbs the marginal weights given the huge combo space).
    nuisance: list[dict[str, str]] = []
    seen_combo: set[tuple[str, ...]] = set()
    attempts = 0
    while len(nuisance) < n_per_class and attempts < n_per_class * 200:
        attempts += 1
        combo = (
            _weighted(rng, PLATE_STYLES), _weighted(rng, PLATE_SHAPES),
            _weighted(rng, TABLE_SURFACES), _weighted(rng, CAMERA_ANGLES),
            _weighted(rng, CUTLERY_STATES), _weighted(rng, LIGHTINGS),
        )
        if combo not in seen_combo:
            seen_combo.add(combo)
            nuisance.append(dict(style=combo[0], shape=combo[1], surface=combo[2],
                                 angle=combo[3], cutlery=combo[4], lighting=combo[5]))

    # 2) Each content class fills the SAME nuisance slots; only {contents} varies. Every
    #    prompt is unique because its nuisance combo is unique (even if content repeats).
    prompts: dict[str, list[str]] = {}
    for class_name in CONTENT_CLASSES:
        out = []
        for nu in nuisance:
            prompt = PROMPT_TEMPLATE.format(contents=_make_contents(rng, class_name), **nu)
            out.append(prompt[0].upper() + prompt[1:])  # capitalize the sentence start
        prompts[class_name] = out

    # 3) `unclassified` borrows random prompts from the content classes; step 03 then
    #    corrupts these normal-looking plates until the state is unreadable.
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


# ---------------------------------------------------------------------------
# Eat-down EDIT instruction (for the image-editing leftovers pipeline).
# Unlike the text-to-image prompts above, this is an INSTRUCTION given to an
# image+text editing model (SDXL-InstructPix2Pix) to take a full plate and make
# it look mostly eaten (~a quarter left). Shared by the edit_leftovers_* scripts.
# ---------------------------------------------------------------------------
EAT_INSTRUCTION_AMOUNTS = [
    "only about a quarter of the food left",
    "roughly a fifth of the meal remaining",
    "just a few small scraps of food left",
    "only a small amount of food remaining in one corner",
]


def build_eat_instruction(rng: random.Random) -> str:
    """Randomized 'eat-down' edit instruction (~1/2 of plates get cutlery, ~1/4 a napkin)."""
    parts = [
        f"Make the plate look like a person has eaten most of the meal, leaving "
        f"{rng.choice(EAT_INSTRUCTION_AMOUNTS)}:",
        "scattered messy scraps to one side, smeared sauce, scattered crumbs and bite marks, with "
        "the rest of the plate empty and bare.",
    ]
    parts.append("Leave a dirty fork or spoon resting in the plate."
                 if rng.random() < 0.5 else "Leave no cutlery on the plate.")
    if rng.random() < 0.25:
        parts.append("Leave a crumpled used paper napkin inside the plate.")
    parts.append("Keep the same plate, table, lighting and camera angle.")
    return " ".join(parts)
