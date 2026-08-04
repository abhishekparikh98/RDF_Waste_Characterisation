"""
Train a YOLOv8 detector for multi-object waste image analysis.

Supports transfer learning from the Ultralytics pretrained weights
(default ``yolov8n.pt``). The training loop is driven by the
Ultralytics ``YOLO`` API, but the script wraps the call to add
structured logging, deterministic configuration, and a deterministic
output directory layout that matches the existing project conventions.

The script does not download any dataset. The user is expected to
provide a YOLO-format dataset YAML file (for example, generated from
a TACO or Roboflow export). The default path is
``data/yolo/dataset.yaml``; the path is configurable from the command
line, so this script can be reused for any YOLO-compatible dataset.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(log_file: Path | None = None) -> logging.Logger:
    """Configure console and file logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger("yolo_train")
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
    parser = argparse.ArgumentParser(description="Train a YOLOv8 detector for waste images")
    parser.add_argument(
        "--data",
        type=str,
        default=str(project_root / "data" / "yolo" / "dataset.yaml"),
        help="Path to the YOLO dataset YAML file (must define train/val/test).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics pretrained weights to start from (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (square).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(project_root / "results" / "yolo_runs"),
        help="Directory for Ultralytics training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="waste_yolov8n",
        help="Run name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training (cpu, 0, 0,1, ...).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-best",
        type=str,
        default=str(project_root / "models" / "yolo_best.pt"),
        help="Destination path for the best model checkpoint.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (writes to results/yolo_runs/<name>).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = project_root / "yolo_training.log"
    logger = setup_logging(log_file)
    logger.info("=" * 80)
    logger.info("YOLOv8 Waste Detection Training")
    logger.info("=" * 80)
    logger.info("Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(
            "Dataset YAML not found at %s. Prepare a YOLO-format dataset and "
            "create the YAML file. See data/yolo/README.md for instructions.",
            data_path,
        )
        return

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        logger.error("ultralytics is not installed. Run: pip install ultralytics>=8.0.0")
        raise exc

    save_best_path = Path(args.save_best)
    save_best_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading pretrained weights: %s", args.model)
    model = YOLO(args.model)

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training")
    logger.info("  data: %s", data_path)
    logger.info("  epochs: %d", args.epochs)
    logger.info("  imgsz: %d", args.imgsz)
    logger.info("  batch: %d", args.batch)
    logger.info("  patience: %d", args.patience)
    logger.info("  device: %s", args.device)
    logger.info("  seed: %d", args.seed)
    logger.info("  best ckpt destination: %s", save_best_path)

    results = model.train(  # type: ignore[attr-defined]
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        plots=True,
        save=True,
        verbose=True,
    )

    run_dir = project_dir / args.name
    weights_dir = run_dir / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"

    if best_ckpt.exists():
        import shutil

        shutil.copy2(best_ckpt, save_best_path)
        logger.info("Best checkpoint copied to %s", save_best_path)
    else:
        logger.warning("best.pt not found at %s; using last.pt as fallback", best_ckpt)
        if last_ckpt.exists():
            import shutil

            shutil.copy2(last_ckpt, save_best_path.with_name("yolo_last.pt"))
            logger.info("Last checkpoint copied to %s", save_best_path.with_name("yolo_last.pt"))

    summary_path = project_dir / args.name / "training_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(f"YOLOv8 training run: {args.name}\n")
        handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Data: {data_path}\n")
        handle.write(f"Pretrained weights: {args.model}\n")
        handle.write(f"Epochs: {args.epochs}\n")
        handle.write(f"Image size: {args.imgsz}\n")
        handle.write(f"Batch size: {args.batch}\n")
        handle.write(f"Patience: {args.patience}\n")
        handle.write(f"Device: {args.device}\n")
        handle.write(f"Best checkpoint: {save_best_path}\n")
        handle.write(f"TensorBoard: {'enabled' if args.tensorboard else 'disabled'}\n")

    logger.info("Training complete. Summary written to %s", summary_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger("yolo_train").error("Training failed: %s", exc, exc_info=True)
        raise
