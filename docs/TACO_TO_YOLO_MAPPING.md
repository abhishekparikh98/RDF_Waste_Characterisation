# TACO → Project Class Mapping

## Purpose

TACO (Trash Annotations in Context) defines 60 fine-grained waste
categories; the project requires exactly six class names
(`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) that
match the keys of `MATERIAL_FEATURE_LIBRARY` in
`src/multimodal_inference.py`. This document records the mapping
applied by `scripts/convert_taco_to_yolo.py` so the conversion is
auditable and the dissertation can defend the choice.

## How the mapping was derived

Each TACO category was examined in `annotations.json` and assigned to
one of the six project classes based on the material composition
implied by the category name. Where the parent supercategory was
inconsistent with the actual material (e.g. `Paper cup` is treated by
TACO as supercategory `Carton` but is usually plastic-coated), the
mapping was overridden to reflect the actual material composition.
Categories not assigned to a project class are dropped (the
annotation is excluded from training, but the image is still emitted
if at least one of its annotations maps to a project class).

## The mapping table

| Project class | TACO category_id | TACO category name |
|---|---|---|
| `cardboard` | 13 | Toilet tube |
| `cardboard` | 14 | Other carton |
| `cardboard` | 15 | Egg carton |
| `cardboard` | 16 | Drink carton |
| `cardboard` | 17 | Corrugated carton |
| `cardboard` | 18 | Meal carton |
| `cardboard` | 19 | Pizza box |
| `glass` | 6 | Glass bottle |
| `glass` | 9 | Broken glass |
| `glass` | 23 | Glass cup |
| `glass` | 26 | Glass jar |
| `metal` | 0 | Aluminium foil |
| `metal` | 8 | Metal bottle cap |
| `metal` | 10 | Food Can |
| `metal` | 11 | Aerosol |
| `metal` | 12 | Drink can |
| `metal` | 28 | Metal lid |
| `metal` | 50 | Pop tab |
| `metal` | 51 | Rope & strings |
| `metal` | 52 | Scrap metal |
| `metal` | 53 | Shoe |
| `metal` | 54 | Squeezable tube |
| `paper` | 33 | Normal paper |
| `plastic` | 2 | Aluminium blister pack |
| `plastic` | 3 | Carded blister pack |
| `plastic` | 4 | Other plastic bottle |
| `plastic` | 5 | Clear plastic bottle |
| `plastic` | 7 | Plastic bottle cap |
| `plastic` | 21 | Disposable plastic cup |
| `plastic` | 24 | Other plastic cup |
| `plastic` | 29 | Other plastic |
| `plastic` | 36 | Plastic film |
| `plastic` | 39 | Other plastic wrapper |
| `plastic` | 55 | Plastic straw |
| `plastic` | 56 | Paper straw *(plastic-coated in practice)* |
| `plastic` | 57 | Styrofoam piece |
| `trash` | 1 | Battery |
| `trash` | 20 | Paper cup *(often plastic-coated)* |
| `trash` | 22 | Foam cup |
| `trash` | 25 | Food waste |
| `trash` | 27 | Plastic lid |
| `trash` | 30 | Magazine paper |
| `trash` | 31 | Tissues |
| `trash` | 32 | Wrapping paper |
| `trash` | 34 | Paper bag |
| `trash` | 35 | Plastified paper bag |
| `trash` | 37 | Six pack rings |
| `trash` | 38 | Garbage bag |
| `trash` | 40 | Single-use carrier bag |
| `trash` | 41 | Polypropylene bag |
| `trash` | 42 | Crisp packet |
| `trash` | 43 | Spread tub |
| `trash` | 44 | Tupperware |
| `trash` | 45 | Disposable food container |
| `trash` | 46 | Foam food container |
| `trash` | 47 | Other plastic container |
| `trash` | 48 | Plastic glooves |
| `trash` | 49 | Plastic utensils |
| `trash` | 58 | Unlabeled litter |
| `trash` | 59 | Cigarette |

## Notable overrides

These TACO categories were re-routed away from their
`supercategory` because the material composition implied by the
category name does not match the supercategory:

- **`Paper cup` (id 20)** — TACO places this in `Carton`; we route to
  `trash` because commercial paper cups are usually plastic-coated and
  not recyclable as cardboard.
- **`Foam cup` (id 22)** — TACO places this in `Carton`; we route to
  `trash` because foam is not cardboard.
- **`Food waste` (id 25)** — TACO places this in `Carton`; we route to
  `trash` because food waste is not combustible RDF.
- **`Plastic lid` (id 27)** — TACO places this in `Paper`; we route to
  `trash` because it is plastic, not paper.
- **`Glass cup` (id 23)** and **`Other plastic cup` (id 24)** — TACO
  places both in `Carton`; we route to `glass` and `plastic`
  respectively by their actual material.
- **`Paper straw` (id 56)** — TACO places this in `Plastic`; we keep
  it in `plastic` because modern paper straws are plastic-coated.

## Unmapped categories

The following TACO category IDs have **no** entry in the mapping and
are dropped:

- (none at the time of writing — every TACO category is assigned)

If a future version of TACO adds a new category, it will be silently
dropped from the training set until the mapping is updated.

## Where the mapping lives

The mapping is implemented in
`scripts/convert_taco_to_yolo.py::TACO_CATEGORY_TO_PROJECT`. Any
update to the mapping should be reflected both in the script and in
the table above.