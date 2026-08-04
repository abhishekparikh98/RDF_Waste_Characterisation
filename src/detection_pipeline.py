"""
Multi-object detection pipeline.

This module joins the Ultralytics YOLOv8 detector with the existing
material feature library and the existing Random Forest. It is a
plug-in replacement for :class:`MultimodalInferencePipeline` in the
sense that it consumes the same Random Forest and the same
``MATERIAL_FEATURE_LIBRARY``; the only difference is that the visual
stage now returns a list of detected objects instead of a single class.

The output is intentionally compatible with the existing Flask
application: a single ``DetectionInferenceResult`` is returned, with
a ``detections`` list that contains one ``ObjectPrediction`` per
detected bounding box.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from src.multimodal_inference import MATERIAL_FEATURE_LIBRARY, MaterialFeatureMapper
from src.yolo_detector import Detection, DetectionResult, YOLODetector, YOLODetectorConfig


@dataclass(frozen=True)
class ObjectPrediction:
    """The per-object prediction row shown in the Flask UI."""

    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    material_features: Dict[str, Any]
    rdf_suitability: int
    rdf_probability: float
    rdf_label: str


@dataclass(frozen=True)
class DetectionInferenceResult:
    """Structured output for the multi-object inference pipeline."""

    image_path: str
    image_width: int
    image_height: int
    detections: List[ObjectPrediction] = field(default_factory=list)

    @property
    def num_objects(self) -> int:
        return len(self.detections)


class RandomForestRDFPredictor:
    """Wrap the persisted Random Forest ``Pipeline`` for reuse."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Random Forest model not found: {self.model_path}")
        self.model = joblib.load(str(self.model_path))

    def predict(self, features: pd.DataFrame) -> tuple[int, float]:
        suitability = int(self.model.predict(features)[0])
        probability = 1.0
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(features)[0][1])
        return suitability, probability


class DetectionPipeline:
    """Chain the YOLO detector, the material feature library, and the Random Forest.

    The pipeline is the multi-object counterpart of
    :class:`MultimodalInferencePipeline`. It shares the same
    ``MaterialFeatureMapper`` (so the same column schema is used) and
    the same Random Forest joblib artefact.
    """

    def __init__(
        self,
        detector: YOLODetector,
        rdf_model_path: Path,
        feature_mapper: Optional[MaterialFeatureMapper] = None,
    ) -> None:
        self.detector = detector
        self.rdf_predictor = RandomForestRDFPredictor(rdf_model_path)
        self.feature_mapper = feature_mapper or MaterialFeatureMapper()

    def _predict_for_class(self, class_name: str, confidence: float, bbox: tuple[float, float, float, float]) -> ObjectPrediction:
        """Run the Random Forest for a single detected class."""
        features = self.feature_mapper.build_features(class_name)
        suitability, probability = self.rdf_predictor.predict(features)
        return ObjectPrediction(
            class_name=class_name,
            confidence=confidence,
            bbox_xyxy=bbox,
            material_features=features.iloc[0].to_dict(),
            rdf_suitability=suitability,
            rdf_probability=probability,
            rdf_label="Suitable" if suitability == 1 else "Not Suitable",
        )

    def infer(self, image_path: str | Path) -> DetectionInferenceResult:
        """Run the full multi-object inference for one image."""
        image_path = Path(image_path)
        detection_result: DetectionResult = self.detector.detect(image_path)
        predictions: List[ObjectPrediction] = [
            self._predict_for_class(
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox_xyxy,
            )
            for det in detection_result.detections
        ]
        return DetectionInferenceResult(
            image_path=str(image_path),
            image_width=detection_result.image_width,
            image_height=detection_result.image_height,
            detections=predictions,
        )

    def infer_batch(self, image_paths: List[str | Path]) -> List[DetectionInferenceResult]:
        """Run inference over multiple images."""
        return [self.infer(path) for path in image_paths]


def build_default_pipeline(
    yolo_model_path: Path,
    rdf_model_path: Path,
    confidence_threshold: float = 0.45,
    iou_threshold: float = 0.45,
    image_size: int = 640,
    device: str = "cpu",
) -> DetectionPipeline:
    """Construct a :class:`DetectionPipeline` with sane defaults."""
    detector = YOLODetector(
        YOLODetectorConfig(
            model_path=Path(yolo_model_path),
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            image_size=image_size,
            device=device,
        )
    )
    return DetectionPipeline(
        detector=detector,
        rdf_model_path=Path(rdf_model_path),
    )
