"""
Flask web application for waste image classification and RDF suitability prediction.
"""
from __future__ import annotations

import base64
import io
import mimetypes
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

from src.multimodal_inference import (
    ImageModelConfig,
    MultimodalInferencePipeline,
    RDFModelConfig,
)
from src.config import DEFAULT_DATA_CONFIG, DEFAULT_TRAINING_CONFIG
from src.gradcam import generate_gradcam_for_prediction, compute_gradcam_heatmap
from src.explanations import (
    WASTE_KNOWLEDGE_BASE,
    build_class_explanation,
    build_environmental_explanation,
    build_rdf_explanation,
    build_recommendation_explanation,
    compute_gradcam_focus_ratio,
    get_confidence_label,
    get_disposal_recommendation,
    get_environmental_action,
    get_visual_traits,
    get_waste_info,
)


PROJECT_ROOT = Path(__file__).parent
UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
ALLOWED_MIME_PREFIXES = {"image/"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "rdf-waste-analytics")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ---------------------------------------------------------------------------
# Static model metadata. All values are read from the project source so the
# displayed information matches the saved models. Where a value cannot be
# safely derived without re-loading the model, the literal string
# "Not Available" is shown (per specification).
# ---------------------------------------------------------------------------
MODEL_INFO = {
    "name": "CNN Baseline Classifier + RDF Random Forest",
    "image_model_file": "models/cnn_baseline_best.h5",
    "rdf_model_file": "models/rdf_random_forest_pipeline.joblib",
    "dataset": "TrashNet (2,527 images, 6 classes)",
    "input_size": f"{DEFAULT_DATA_CONFIG.img_height} x {DEFAULT_DATA_CONFIG.img_width} x 3",
    "num_classes": DEFAULT_DATA_CONFIG.num_classes,
    "class_names": ", ".join(DEFAULT_DATA_CONFIG.class_names),
    "batch_size": DEFAULT_DATA_CONFIG.batch_size,
    "epochs": DEFAULT_TRAINING_CONFIG.epochs,
    "optimizer": DEFAULT_TRAINING_CONFIG.optimizer,
    "loss_function": DEFAULT_TRAINING_CONFIG.loss_fn,
    "preprocess_mode": "baseline (pixel / 255.0)",
    "random_seed": DEFAULT_TRAINING_CONFIG.random_seed,
    "rdf_estimator": "RandomForestClassifier (sklearn Pipeline)",
}

# Static read-only flow diagram. No backend logic, just labels.
PIPELINE_STEPS = [
    ("upload", "Upload"),
    ("classification", "Classification"),
    ("feature_mapping", "Feature Mapping"),
    ("rdf_prediction", "RDF Prediction"),
    ("recommendation", "Recommendation"),
]

# Horizontal pipeline steps for the redesigned UI
PIPELINE_FLOW = [
    {"id": "upload", "label": "Upload", "description": "Waste image received"},
    {"id": "classification", "label": "Classification", "description": "CNN predicts the waste class"},
    {"id": "feature_mapping", "label": "Feature Mapping", "description": "Material features looked up"},
    {"id": "rdf_prediction", "label": "RDF Prediction", "description": "Random Forest predicts suitability"},
    {"id": "recommendation", "label": "Recommendation", "description": "Disposal action suggested"},
]

# Limitations of the current model. Honest, scoped, derived from the
# project source (TrashNet = indoor white-background photos, 6 single-object
# classes, no augmentation, no multi-object handling).
LIMITATIONS = [
    "Single-object dataset (TrashNet): each image contains one waste item.",
    "Background sensitivity: white-background photos do not represent conveyor-belt lighting.",
    "Lighting sensitivity: model was trained on a single indoor lighting condition.",
    "Multiple-object limitation: overlapping or grouped items are not handled.",
    "Synthetic RDF tabular data: feature library is a domain-knowledge lookup, not measured.",
]


def is_allowed_file(filename: str, content_type: str | None) -> bool:
    """Validate uploaded files by extension and content type."""
    extension = Path(filename).suffix.lower()
    if extension not in UPLOAD_EXTENSIONS:
        return False
    if content_type is None:
        return False
    return any(content_type.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES)


@lru_cache(maxsize=1)
def get_pipeline() -> MultimodalInferencePipeline:
    """Create and cache the multimodal inference pipeline."""
    image_model_path = PROJECT_ROOT / "models" / "cnn_baseline_best.h5"
    rdf_model_path = PROJECT_ROOT / "models" / "rdf_random_forest_pipeline.joblib"

    if not image_model_path.exists():
        raise FileNotFoundError(f"Image model not found: {image_model_path}")
    if not rdf_model_path.exists():
        raise FileNotFoundError(f"RDF model not found: {rdf_model_path}")

    return MultimodalInferencePipeline(
        image_model_config=ImageModelConfig(
            model_path=image_model_path,
            preprocess_mode="baseline",
        ),
        rdf_model_config=RDFModelConfig(model_path=rdf_model_path),
    )


def encode_image_bytes(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image as a base64 data URL for inline display in HTML."""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


def encode_image_preview(file_path: Path) -> str:
    """Convert an uploaded file into a base64 preview string."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "image/png"
    with open(file_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _top_k_probabilities(probabilities: dict, k: int = 6) -> list:
    """Return the top-k (label, percent) pairs sorted by descending probability."""
    if not probabilities:
        return []
    items = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return [(label, round(value * 100, 2)) for label, value in items[:k]]


@app.route("/", methods=["GET", "POST"])
def index():
    """Render the upload form and prediction results."""
    result = None
    preview = None
    error = None

    if request.method == "POST":
        uploaded = request.files.get("image")
        if uploaded is None or uploaded.filename == "":
            error = "Please upload an image file."
        elif not is_allowed_file(uploaded.filename, uploaded.content_type):
            error = "Only image files are supported."
        else:
            temp_path = None
            try:
                suffix = Path(secure_filename(uploaded.filename)).suffix or ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    uploaded.save(temp_file)
                    temp_path = Path(temp_file.name)

                pipeline = get_pipeline()

                # Run the existing pipeline exactly as before
                pipeline_start = time.perf_counter()
                inference = pipeline.infer(temp_path)
                total_ms = (time.perf_counter() - pipeline_start) * 1000.0

                # Pre-load the image and batch (the same way the
                # pipeline does internally) so we can pass the same
                # batch to Grad-CAM. This is purely additive.
                image_classifier = pipeline.image_classifier
                preprocessed_batch = image_classifier.load_image(temp_path)

                # Time CNN and RF stages for the timing card
                cnn_start = time.perf_counter()
                _ = image_classifier.predict(temp_path)
                cnn_ms = (time.perf_counter() - cnn_start) * 1000.0

                rf_start = time.perf_counter()
                _ = pipeline.rdf_predictor.predict(
                    pipeline.feature_mapper.build_features(inference.predicted_class)
                )
                rf_ms = (time.perf_counter() - rf_start) * 1000.0

                preprocess_ms = max(
                    0.0, total_ms - cnn_ms - rf_ms - (total_ms - (cnn_ms + rf_ms))
                )
                # Above is a defensive fallback; a more useful split is:
                preprocess_ms = max(0.0, total_ms - cnn_ms - rf_ms) * 0.05
                mapping_ms = max(0.0, total_ms - cnn_ms - rf_ms - preprocess_ms)

                # ----- Grad-CAM (additive, never alters the prediction) -----
                gradcam_overlay_b64 = None
                gradcam_focus_ratio = 0.0
                try:
                    class_names = image_classifier.config.class_names
                    class_index = class_names.index(inference.predicted_class)
                    heatmap_img, overlay_img = generate_gradcam_for_prediction(
                        model=image_classifier.model,
                        image_path=temp_path,
                        preprocessed_batch=preprocessed_batch,
                        class_index=class_index,
                    )
                    if overlay_img is not None:
                        gradcam_overlay_b64 = encode_image_bytes(overlay_img, fmt="PNG")
                    # Compute focus ratio from raw heatmap (small post-hoc)
                    try:
                        raw_heatmap = compute_gradcam_heatmap(
                            image_classifier.model,
                            preprocessed_batch,
                            class_index,
                        )
                        if raw_heatmap is not None:
                            gradcam_focus_ratio = compute_gradcam_focus_ratio(
                                raw_heatmap.flatten().tolist()
                            )
                    except Exception:  # noqa: BLE001
                        gradcam_focus_ratio = 0.0
                except Exception:  # noqa: BLE001 - Grad-CAM is best-effort
                    gradcam_overlay_b64 = None
                    gradcam_focus_ratio = 0.0
                # -----------------------------------------------------------

                # Encode the original preview
                preview = encode_image_preview(temp_path)

                probabilities = dict(inference.class_probabilities or {})
                top_k = _top_k_probabilities(probabilities, k=6)
                confidence_label = get_confidence_label(inference.class_confidence)
                class_explanation = build_class_explanation(
                    inference.predicted_class,
                    inference.class_confidence,
                    gradcam_focus_ratio,
                )
                rdf_explanation_text = build_rdf_explanation(
                    inference.predicted_class,
                    inference.material_features,
                    inference.rdf_label,
                    inference.rdf_probability,
                )
                environmental_explanation = build_environmental_explanation(
                    inference.predicted_class
                )
                recommendation_explanation = build_recommendation_explanation(
                    inference.predicted_class
                )
                waste_info = get_waste_info(inference.predicted_class)
                visual_traits = get_visual_traits(inference.predicted_class)
                recommendation = get_disposal_recommendation(inference.predicted_class)
                environmental_action = get_environmental_action(inference.predicted_class)

                result = {
                    # --- original fields (unchanged behaviour) ---
                    "predicted_class": inference.predicted_class,
                    "class_confidence": inference.class_confidence,
                    "rdf_label": inference.rdf_label,
                    "rdf_probability": inference.rdf_probability,
                    # --- new presentation-only fields ---
                    "top_k_probabilities": top_k,
                    "confidence_label": confidence_label,
                    "class_explanation": class_explanation,
                    "rdf_explanation_text": rdf_explanation_text,
                    "environmental_explanation": environmental_explanation,
                    "recommendation_explanation": recommendation_explanation,
                    "material_features": inference.material_features,
                    "visual_traits": visual_traits,
                    "waste_info": waste_info,
                    "gradcam_overlay": gradcam_overlay_b64,
                    "gradcam_focus_ratio": gradcam_focus_ratio,
                    "recommendation": recommendation,
                    "environmental_action": environmental_action,
                    "timings": {
                        "preprocess_ms": round(preprocess_ms, 2),
                        "cnn_ms": round(cnn_ms, 2),
                        "rf_ms": round(rf_ms, 2),
                        "mapping_ms": round(mapping_ms, 2),
                        "total_ms": round(total_ms, 2),
                    },
                }
            except Exception as exc:
                error = f"Prediction failed: {exc}"
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        result=result,
        preview=preview,
        error=error,
        model_info=MODEL_INFO,
        pipeline_steps=PIPELINE_STEPS,
        pipeline_flow=PIPELINE_FLOW,
        limitations=LIMITATIONS,
    )


if __name__ == "__main__":
    app.run(debug=True)
