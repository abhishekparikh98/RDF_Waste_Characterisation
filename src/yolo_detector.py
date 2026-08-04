"""
YOLOv8 object detection wrapper for multi-object waste image analysis.

This module replaces the single-label CNN classifier with a multi-object
detector. The detector returns a list of bounding boxes, each tagged with
a class name and a confidence score, so the downstream pipeline can
reason about multiple waste items in a single image.

Only the detection stage is new. The material feature library and the
Random Forest are reused unchanged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class YOLODetectorConfig:
    """Configuration for the trained YOLOv8 detector."""

    model_path: Path
    class_names: Optional[List[str]] = None
    # Threshold raised from the standard 0.25 to 0.45 because the trained
    # detector reaches only mAP@0.5 ~= 0.20 (see reports/yolo_evaluation_report.md).
    # At 0.25 the model emits many false-positive, over-confident predictions on
    # out-of-distribution photos (e.g. real plastic bottles mislabelled as
    # "trash"). 0.45 suppresses those while still allowing genuine detections
    # through. This is a standard precision/recall trade-off documented in
    # the YOLOv8 evaluation methodology (COCO / Lin et al., 2014).
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.class_names is not None:
            object.__setattr__(self, "class_names", list(self.class_names))


@dataclass(frozen=True)
class Detection:
    """A single detected object bounding box."""

    class_name: str
    confidence: float
    bbox_xyxy: Tuple[float, float, float, float]  # x1, y1, x2, y2
    class_index: int = 0


@dataclass(frozen=True)
class DetectionResult:
    """Structured output from the YOLO detector."""

    image_path: str
    image_width: int
    image_height: int
    detections: List[Detection] = field(default_factory=list)


class YOLODetector:
    """Load and run inference with a trained YOLOv8 detector.

    The class is a thin wrapper around the Ultralytics YOLO class. It
    exposes a single ``detect`` method that returns a ``DetectionResult``
    dataclass so the rest of the project can consume the detector
    without importing Ultralytics directly.
    """

    def __init__(self, config: YOLODetectorConfig) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Ultralytics YOLO model from disk."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLODetector. "
                "Install with: pip install ultralytics>=8.0.0"
            ) from exc

        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {self.config.model_path}. "
                "Train one with scripts/train_yolo.py or download a pretrained "
                "checkpoint."
            )

        logger.info("Loading YOLO model from %s", self.config.model_path)
        self._model = YOLO(str(self.config.model_path))
        self._class_names = self._resolve_class_names()

    def _resolve_class_names(self) -> List[str]:
        """Return the class names from the loaded model, with safe fallback."""
        if self.config.class_names is not None:
            return self.config.class_names
        names = getattr(self._model, "names", None)
        if names is None:
            return []
        if isinstance(names, dict):
            return [names[k] for k in sorted(names.keys())]
        return list(names)

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

    def detect(self, image_path: str | Path) -> DetectionResult:
        """Run detection on a single image and return a DetectionResult."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as img:
            width, height = img.size

        results = self._model.predict(  # type: ignore[union-attr]
            source=str(image_path),
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return DetectionResult(
                image_path=str(image_path),
                image_width=width,
                image_height=height,
                detections=detections,
            )

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                image_path=str(image_path),
                image_width=width,
                image_height=height,
                detections=detections,
            )

        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls).astype(int)

        for box, conf, cls_id in zip(xyxy, confs, cls_ids):
            x1, y1, x2, y2 = (float(v) for v in box)
            class_name = (
                self._class_names[cls_id]
                if 0 <= cls_id < len(self._class_names)
                else str(cls_id)
            )
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox_xyxy=(x1, y1, x2, y2),
                    class_index=int(cls_id),
                )
            )

        logger.debug("Detected %d objects in %s", len(detections), image_path)
        return DetectionResult(
            image_path=str(image_path),
            image_width=width,
            image_height=height,
            detections=detections,
        )

    def detect_batch(self, image_paths: List[str | Path]) -> List[DetectionResult]:
        """Run detection on a list of images."""
        return [self.detect(path) for path in image_paths]
