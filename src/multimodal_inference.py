"""
End-to-end multimodal inference for RDF suitability prediction.

Workflow:
Image -> Waste Classification -> Material Features -> RDF Suitability Prediction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

from src.config import DEFAULT_DATA_CONFIG


@dataclass(frozen=True)
class ImageModelConfig:
    """Configuration for the trained image classifier."""

    model_path: Path
    class_names: List[str] = None
    input_shape: tuple[int, int] = (224, 224)
    preprocess_mode: str = "baseline"

    def __post_init__(self):
        if self.class_names is None:
            object.__setattr__(self, "class_names", list(DEFAULT_DATA_CONFIG.class_names))


@dataclass(frozen=True)
class RDFModelConfig:
    """Configuration for the trained RDF suitability model."""

    model_path: Path


@dataclass(frozen=True)
class InferenceResult:
    """Structured output from the multimodal pipeline."""

    image_path: str
    predicted_class: str
    class_confidence: float
    material_features: Dict[str, object]
    rdf_suitability: int
    rdf_probability: float
    rdf_label: str
    class_probabilities: Dict[str, float] = field(default_factory=dict)


MATERIAL_FEATURE_LIBRARY: Dict[str, Dict[str, object]] = {
    "cardboard": {
        "material_type": "cardboard",
        "moisture_content": 12.0,
        "contamination_level": 1.5,
        "combustibility": 8.6,
        "calorific_value": 16.5,
    },
    "glass": {
        "material_type": "glass",
        "moisture_content": 1.0,
        "contamination_level": 1.2,
        "combustibility": 0.0,
        "calorific_value": 0.0,
    },
    "metal": {
        "material_type": "metal",
        "moisture_content": 1.0,
        "contamination_level": 1.5,
        "combustibility": 0.2,
        "calorific_value": 0.1,
    },
    "paper": {
        "material_type": "paper",
        "moisture_content": 14.0,
        "contamination_level": 2.0,
        "combustibility": 8.0,
        "calorific_value": 15.0,
    },
    "plastic": {
        "material_type": "plastic",
        "moisture_content": 2.0,
        "contamination_level": 2.5,
        "combustibility": 9.0,
        "calorific_value": 38.0,
    },
    "trash": {
        "material_type": "organic",
        "moisture_content": 60.0,
        "contamination_level": 6.0,
        "combustibility": 4.0,
        "calorific_value": 6.0,
    },
}


class ImageClassifier:
    """Load and run inference with the trained image classifier."""

    def __init__(self, config: ImageModelConfig):
        self.config = config
        self.model = keras.models.load_model(
            str(config.model_path),
            custom_objects={"BaselineCNN": keras.Sequential},
        )
        self.preprocess_fn = self._resolve_preprocess_fn(config.preprocess_mode)

    def _resolve_preprocess_fn(self, mode: str) -> Callable[[np.ndarray], np.ndarray]:
        if mode == "mobilenetv2":
            return mobilenet_preprocess_input
        if mode == "resnet50":
            return resnet_preprocess_input
        return lambda images: images / 255.0

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess a single image for inference."""
        image = Image.open(image_path).convert("RGB")
        # BILINEAR matches the TensorFlow bilinear used by
        # keras.preprocessing.image_dataset_from_directory during training,
        # which keeps train/serve preprocessing numerically consistent.
        image = image.resize(self.config.input_shape, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)
        return self.preprocess_fn(array)

    def predict(self, image_path: Path) -> tuple[str, float, np.ndarray]:
        """Predict the waste class and confidence."""
        batch = self.load_image(image_path)
        probabilities = self.model.predict(batch, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        predicted_class = self.config.class_names[class_index]
        return predicted_class, confidence, probabilities


class MaterialFeatureMapper:
    """Convert predicted waste classes into RDF material features."""

    def __init__(self, feature_library: Dict[str, Dict[str, object]] | None = None):
        self.feature_library = feature_library or MATERIAL_FEATURE_LIBRARY

    def build_features(self, waste_class: str) -> pd.DataFrame:
        """Create a single-row feature frame for the RDF model."""
        if waste_class not in self.feature_library:
            waste_class = "trash"
        return pd.DataFrame([self.feature_library[waste_class]])


class RDFSuitabilityPredictor:
    """Load and run inference with the trained RDF Random Forest pipeline."""

    def __init__(self, config: RDFModelConfig):
        self.config = config
        self.model = joblib.load(str(config.model_path))

    def predict(self, features: pd.DataFrame) -> tuple[int, float]:
        """Predict RDF suitability and probability."""
        suitability = int(self.model.predict(features)[0])
        probability = 1.0
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(features)[0][1])
        return suitability, probability


class MultimodalInferencePipeline:
    """Chain the image classifier, material feature mapping, and RDF predictor."""

    def __init__(
        self,
        image_model_config: ImageModelConfig,
        rdf_model_config: RDFModelConfig,
        feature_mapper: MaterialFeatureMapper | None = None,
    ) -> None:
        self.image_classifier = ImageClassifier(image_model_config)
        self.rdf_predictor = RDFSuitabilityPredictor(rdf_model_config)
        self.feature_mapper = feature_mapper or MaterialFeatureMapper()

    def infer(self, image_path: str | Path) -> InferenceResult:
        """Run the full multimodal inference workflow for one image."""
        image_path = Path(image_path)
        predicted_class, confidence, probabilities = self.image_classifier.predict(image_path)
        material_features = self.feature_mapper.build_features(predicted_class)
        rdf_label, rdf_probability = self.rdf_predictor.predict(material_features)

        # Map the softmax vector onto class names so the UI can rank the top-k
        # without re-running the model. Existing fields are unchanged.
        class_probabilities = {
            name: float(probabilities[index])
            for index, name in enumerate(self.image_classifier.config.class_names)
        }

        return InferenceResult(
            image_path=str(image_path),
            predicted_class=predicted_class,
            class_confidence=confidence,
            material_features=material_features.iloc[0].to_dict(),
            rdf_suitability=rdf_label,
            rdf_probability=rdf_probability,
            rdf_label="Suitable" if rdf_label == 1 else "Not Suitable",
            class_probabilities=class_probabilities,
        )

    def infer_batch(self, image_paths: List[str | Path]) -> List[InferenceResult]:
        """Run inference over multiple images."""
        return [self.infer(path) for path in image_paths]
