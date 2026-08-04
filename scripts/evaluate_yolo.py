"""
Evaluate a trained YOLOv8 detector on a held-out YOLO-format dataset.

Generates:

- Precision, recall, mAP50, mAP50-95 (overall and per-class)
- A normalised confusion matrix (PNG)
- A Markdown evaluation report
- Sampled prediction visualisations (PNG)

The script does not download any dataset. The dataset YAML path is
configurable; the default is ``data/yolo/dataset.yaml``.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger("yolo_eval")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    try:
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (AttributeError, UnicodeDecodeError):
        pass

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 detector")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(project_root / "models" / "yolo_best.pt"),
        help="Path to the trained YOLOv8 weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(project_root / "data" / "yolo" / "dataset.yaml"),
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for evaluation.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for inference.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for evaluation (cpu, 0, 0,1, ...).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of prediction sample images to visualise.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(project_root / "results"),
        help="Directory for output PNGs and JSON metrics.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=str(project_root / "reports"),
        help="Directory for the Markdown report.",
    )
    return parser.parse_args()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """Plot a normalised confusion matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = np.divide(
            cm.astype(float) * 100.0,
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("YOLOv8 Confusion Matrix (Normalised %)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_pct[i, j]
            colour = "white" if value > 50 else "black"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=colour, fontsize=9)
    fig.colorbar(im, ax=ax, label="Percentage (%)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_per_class_table(metrics: Any) -> List[List[str]]:
    """Convert Ultralytics per-class metrics into a Markdown table."""
    rows: List[List[str]] = []
    names = list(metrics.names.values()) if hasattr(metrics, "names") else []
    for i, name in enumerate(names):
        precision = float(metrics.precision[i]) if metrics.precision is not None else 0.0
        recall = float(metrics.recall[i]) if metrics.recall is not None else 0.0
        map50 = float(metrics.map50[i]) if metrics.map50 is not None else 0.0
        map5095 = float(metrics.map[i]) if metrics.map is not None else 0.0
        rows.append([name, f"{precision:.4f}", f"{recall:.4f}", f"{map50:.4f}", f"{map5095:.4f}"])
    return rows


def save_prediction_samples(
    predictor: Any,
    val_images: List[str],
    save_dir: Path,
    num_samples: int,
) -> List[Path]:
    """Run inference on a handful of validation images and save annotated outputs."""
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for idx, image_path in enumerate(val_images[:num_samples]):
        try:
            results = predictor.predict(source=image_path, save=False, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("yolo_eval").warning("Failed to predict %s: %s", image_path, exc)
            continue
        if not results:
            continue
        result = results[0]
        annotated = result.plot()
        output_path = save_dir / f"sample_{idx:02d}.png"
        try:
            import cv2  # type: ignore

            cv2.imwrite(str(output_path), annotated)
        except Exception:  # noqa: BLE001
            from PIL import Image

            Image.fromarray(annotated[..., ::-1]).save(output_path)
        saved.append(output_path)
    return saved


def write_report(
    overall: Dict[str, float],
    per_class_rows: List[List[str]],
    class_names: List[str],
    confusion_matrix_path: Path,
    sample_paths: List[Path],
    weights_path: Path,
    data_path: Path,
    report_path: Path,
) -> None:
    """Write the Markdown evaluation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# YOLOv8 Waste Detection - Evaluation Report\n\n")
        handle.write(f"**Generated:** {timestamp}  \n")
        handle.write(f"**Weights:** `{weights_path}`  \n")
        handle.write(f"**Dataset:** `{data_path}`\n\n")
        handle.write("---\n\n")
        handle.write("## Overall Metrics\n\n")
        handle.write("| Metric | Value |\n|---|---:|\n")
        handle.write(f"| Precision | {overall['precision']:.4f} |\n")
        handle.write(f"| Recall | {overall['recall']:.4f} |\n")
        handle.write(f"| mAP50 | {overall['map50']:.4f} |\n")
        handle.write(f"| mAP50-95 | {overall['map']:.4f} |\n\n")
        handle.write("## Per-Class Metrics\n\n")
        handle.write("| Class | Precision | Recall | mAP50 | mAP50-95 |\n|---|---:|---:|---:|---:|\n")
        for row in per_class_rows:
            handle.write("| " + " | ".join(row) + " |\n")
        handle.write("\n## Confusion Matrix\n\n")
        handle.write(f"Normalised confusion matrix: `{confusion_matrix_path.relative_to(project_root)}`\n\n")
        handle.write("## Prediction Samples\n\n")
        if sample_paths:
            for path in sample_paths:
                handle.write(f"- `{path.relative_to(project_root)}`\n")
        else:
            handle.write("- No prediction samples were generated.\n")
        handle.write("\n## Notes\n\n")
        handle.write("- All metrics are computed on the validation split defined by the dataset YAML.\n")
        handle.write("- The confusion matrix is normalised by row (true class).\n")
        handle.write("- Sample visualisations are saved to `results/yolo_predictions/`.\n")


def main() -> None:
    args = parse_args()
    log_file = project_root / "yolo_evaluation.log"
    logger = setup_logging(log_file)
    logger.info("=" * 80)
    logger.info("YOLOv8 Evaluation")
    logger.info("=" * 80)

    weights_path = Path(args.weights)
    data_path = Path(args.data)
    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        logger.error("Weights not found at %s", weights_path)
        return
    if not data_path.exists():
        logger.error("Dataset YAML not found at %s", data_path)
        return

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        logger.error("ultralytics is not installed. Run: pip install ultralytics>=8.0.0")
        raise exc

    logger.info("Loading model from %s", weights_path)
    model = YOLO(str(weights_path))

    logger.info("Running validation")
    val_metrics = model.val(  # type: ignore[attr-defined]
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,
        save_json=True,
        project=str(results_dir / "yolo_runs"),
        name="val",
        exist_ok=True,
        verbose=True,
    )

    metrics = val_metrics.box
    overall = {
        "precision": float(np.mean(metrics.precision)) if metrics.precision is not None else 0.0,
        "recall": float(np.mean(metrics.recall)) if metrics.recall is not None else 0.0,
        "map50": float(np.mean(metrics.map50)) if metrics.map50 is not None else 0.0,
        "map": float(np.mean(metrics.map)) if metrics.map is not None else 0.0,
    }
    logger.info("Overall: precision=%.4f recall=%.4f mAP50=%.4f mAP50-95=%.4f",
                overall["precision"], overall["recall"], overall["map50"], overall["map"])

    class_names = list(metrics.names.values()) if hasattr(metrics, "names") else []
    per_class_rows = build_per_class_table(metrics)

    cm_array = None
    if hasattr(val_metrics, "confusion_matrix") and hasattr(val_metrics.confusion_matrix, "matrix"):
        cm_array = np.asarray(val_metrics.confusion_matrix.matrix)
    if cm_array is None or cm_array.size == 0:
        cm_array = np.zeros((len(class_names), len(class_names)), dtype=int)

    cm_path = results_dir / "yolo_confusion_matrix.png"
    plot_confusion_matrix(cm_array, class_names, cm_path)
    logger.info("Confusion matrix saved to %s", cm_path)

    metrics_path = results_dir / "yolo_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "overall": overall,
                "per_class": [
                    {"class": row[0], "precision": row[1], "recall": row[2], "map50": row[3], "map50_95": row[4]}
                    for row in per_class_rows
                ],
                "weights": str(weights_path),
                "data": str(data_path),
            },
            handle,
            indent=2,
        )
    logger.info("Metrics JSON saved to %s", metrics_path)

    samples_dir = results_dir / "yolo_predictions"
    sample_paths: List[Path] = []
    val_images: List[str] = []
    try:
        val_dir = data_path.parent / "val" / "images"
        if val_dir.exists():
            val_images = [str(p) for p in sorted(val_dir.glob("*")) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    except Exception:  # noqa: BLE001
        val_images = []

    if val_images:
        sample_paths = save_prediction_samples(model, val_images, samples_dir, args.num_samples)
        logger.info("Saved %d prediction samples to %s", len(sample_paths), samples_dir)
    else:
        logger.info("No validation images found for sample visualisation.")

    report_path = reports_dir / "yolo_evaluation_report.md"
    write_report(
        overall=overall,
        per_class_rows=per_class_rows,
        class_names=class_names,
        confusion_matrix_path=cm_path,
        sample_paths=sample_paths,
        weights_path=weights_path,
        data_path=data_path,
        report_path=report_path,
    )
    logger.info("Report written to %s", report_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger("yolo_eval").error("Evaluation failed: %s", exc, exc_info=True)
        raise
