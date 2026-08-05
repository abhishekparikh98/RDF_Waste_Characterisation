# TACO → Project Class Mapping

The full 60-to-6 mapping table lives in code, not in this document, to keep a
single source of truth between the mapping dictionary and the preprocessing
output.

- **Source of truth**: `scripts/convert_taco_to_yolo.py` — the
  `TACO_CATEGORY_TO_PROJECT: Dict[int, str]` constant at lines 64–86 maps
  every TACO `category_id` to one of the 6 project classes.
- **Output schema**: `data/yolo/data.yaml` — the six `names` entries used
  during YOLO training (`cardboard`, `glass`, `metal`, `paper`, `plastic`,
  `trash`).
- **Live mapping**: `scripts/convert_taco_to_yolo.py` is also where the
  mapping is applied at runtime; running it against a fresh TACO annotations
  file regenerates `data/yolo/` with the correct labels.

If you need to add or reclassify a TACO category:

1. Edit `TACO_CATEGORY_TO_PROJECT` in `scripts/convert_taco_to_yolo.py`.
2. Update `PROJECT_CLASSES` in the same file (lines 57 area).
3. Regenerate the dataset with `python scripts/convert_taco_to_yolo.py`.
4. Retrain with `python scripts/train_yolo.py`.
