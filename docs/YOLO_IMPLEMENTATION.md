# YOLO Implementation Notes

## Background

The original MSc project used a single-label CNN classifier
(`BaselineCNN`, with optional MobileNetV2 and ResNet50 transfer
learning) to identify waste classes from images. The teacher-student
demo worked well on TrashNet single-object images, but laboratory
testing on real household waste showed that the CNN could only emit
one class per image, which is unrealistic for conveyor-belt or
kitchen-counter scenes.

This note documents the multi-object upgrade that replaces the CNN
classifier with a YOLOv8 detector while preserving the rest of the
project.

## Why YOLO Replaced the CNN

The CNN was replaced by YOLOv8 for three reasons:

1. **Multi-object detection.** A real waste image contains multiple
   items. A classifier can only return the most likely class for the
   whole image, which is misleading when the image contains several
   distinctive objects. YOLO emits one bounding box per detected
   object, so the system can produce a per-object RDF verdict.
2. **Localisation.** Knowing *where* an object is in the frame is
   useful for downstream robotic or conveyor-belt pickers. The CNN
   classifier only knows the dominant class.
3. **Maturity.** YOLOv8 is a mature, well-supported model family
   with first-class transfer learning. Training a from-scratch CNN
   on 1,763 images is not feasible; fine-tuning YOLOv8n on a
   annotated waste-detection dataset is.

`YOLOv8n` (the nano variant) was chosen as the default backbone
because the project is an MSc prototype and the smaller model is
fast to train and easy to deploy on CPU.

## Architecture Changes

The previous pipeline was:

```
Image -> CNN -> Waste Class -> Material Features -> Random Forest -> RDF verdict
```

The new pipeline is:

```
Image -> YOLOv8 -> List of (class, confidence, bbox) -> Material Features (per object) -> Random Forest (per object) -> List of RDF verdicts
```

The material feature library and the Random Forest are **unchanged**.
The YOLO detector is a plug-in replacement for the CNN inside the
inference pipeline. The Flask application still returns a single
result page; the page now contains a "Detected Objects" card that
lists each detection along with its RDF verdict.

## Training

`scripts/train_yolo.py` drives the fine-tuning. The script is a
thin wrapper around `ultralytics.YOLO.train()` that:

- Accepts a path to a YOLO-format dataset YAML (default
  `data/yolo/dataset.yaml`).
- Starts from the Ultralytics pretrained weights (`yolov8n.pt` by
  default) — this is transfer learning, not from-scratch training.
- Configures the optimiser, learning rate, image size, batch size,
  patience, and seed from the command line.
- Saves the best checkpoint to `models/yolo_best.pt`.
- Optionally enables TensorBoard logging.

The training script does not download any dataset. The user is
expected to provide a YOLO-format dataset YAML. See
`data/yolo/README.md` for the expected layout and recommended
sources (TACO, Roboflow Waste Detection).

## Evaluation

`scripts/evaluate_yolo.py` runs the trained model on the validation
split and produces:

- Precision, recall, mAP50, mAP50-95 (overall and per-class).
- A normalised confusion matrix PNG.
- A Markdown evaluation report at `reports/yolo_evaluation_report.md`.
- A JSON metrics file at `results/yolo_metrics.json`.
- Sample prediction visualisations under `results/yolo_predictions/`.

## Inference

`src/detection_pipeline.py` chains the YOLO detector with the
existing material feature library and the existing Random Forest.
The pipeline is exposed by `app.py` and by the CLI script
`scripts/run_multimodal_inference.py` (the CLI script was not
modified; a new multi-object CLI can be added in future work).

The Flask route now calls `get_yolo_pipeline()` instead of
`get_pipeline()`. The legacy CNN pipeline is still available
through `get_pipeline()` for Grad-CAM demonstrations and for the
existing single-label report.

## Limitations

- The project still relies on the same material feature library.
  Per-object material features are looked up, not measured.
- YOLO inference is slower than the CNN at small batch sizes because
  the model is larger and the NMS post-processing step is non-trivial.
- The current demonstration does not perform on-the-fly annotation
  of the uploaded image. Adding a bounding-box overlay would be a
  useful improvement.
- The class names emitted by YOLOv8 must match the keys in
  `MATERIAL_FEATURE_LIBRARY`. The detector should be trained on a
  dataset that uses the same six class names.

## Reproducibility

- Random seed is 42 throughout the training script.
- The configuration is exposed via command-line arguments, so every
  run can be re-executed with the same arguments.
- The training script writes a summary file to
  `results/yolo_runs/<run-name>/training_summary.txt`.
- The evaluation script writes a JSON metrics file to
  `results/yolo_metrics.json`.
