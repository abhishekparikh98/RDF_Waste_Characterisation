"""
Flask web application for waste image classification and RDF suitability prediction.
"""
from __future__ import annotations

import base64
import io
import mimetypes
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

from src.multimodal_inference import (
    ImageModelConfig,
    MultimodalInferencePipeline,
    RDFModelConfig,
)
from src.detection_pipeline import (
    DetectionInferenceResult,
    DetectionPipeline,
    build_default_pipeline,
)
from src.gradcam import generate_gradcam_for_prediction
from src.explanations import (
    WASTE_KNOWLEDGE_BASE,
    build_class_explanation,
    build_environmental_explanation,
    build_rdf_explanation,
    compute_gradcam_focus_ratio,
    get_confidence_label,
    get_visual_traits,
    get_waste_info,
)


PROJECT_ROOT = Path(__file__).parent
UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
ALLOWED_MIME_PREFIXES = {"image/"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "rdf-waste-analytics")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


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
    """Create and cache the legacy CNN-based multimodal inference pipeline."""
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


@lru_cache(maxsize=1)
def get_yolo_pipeline() -> DetectionPipeline:
    """Create and cache the YOLOv8-based multi-object detection pipeline."""
    yolo_model_path = PROJECT_ROOT / "models" / "yolo_best.pt"
    rdf_model_path = PROJECT_ROOT / "models" / "rdf_random_forest_pipeline.joblib"

    if not yolo_model_path.exists():
        raise FileNotFoundError(
            f"YOLO model not found at {yolo_model_path}. Train one with "
            "scripts/train_yolo.py or drop a pretrained yolov8n waste detection "
            "checkpoint at this path."
        )
    if not rdf_model_path.exists():
        raise FileNotFoundError(f"RDF model not found: {rdf_model_path}")

    return build_default_pipeline(
        yolo_model_path=yolo_model_path,
        rdf_model_path=rdf_model_path,
        # 0.45 instead of the standard 0.25 — see comment in src/yolo_detector.py
        # for the mAP-driven justification. Kept as an explicit kwarg so it is
        # visible in the dissertation code listing.
        confidence_threshold=0.45,
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

                pipeline = get_yolo_pipeline()

                # Run the YOLOv8 detection pipeline (multi-object)
                inference: DetectionInferenceResult = pipeline.infer(temp_path)

                # Always run the legacy CNN in parallel so Grad-CAM is
                # available regardless of whether YOLO found detections.
                # The CNN is a single-label classifier trained on TrashNet,
                # so its Grad-CAM is meaningful even when YOLO is the active
                # detector.  Both models' results are surfaced to the UI.
                legacy_pipeline = get_pipeline()
                legacy_result = legacy_pipeline.infer(temp_path)

                # Compute Grad-CAM from the CNN's prediction. Always run
                # this — if YOLO emitted a detection, the page header still
                # shows the bounding boxes, while the bottom panel shows
                # what the CNN attended to.
                gradcam_overlay_b64 = None
                gradcam_focus_ratio = 0.0
                try:
                    class_names_list = list(legacy_pipeline.image_classifier.config.class_names)
                    cnn_class_idx = class_names_list.index(legacy_result.predicted_class)
                    preprocessed = legacy_pipeline.image_classifier.load_image(temp_path)
                    heatmap_img, overlay_img = generate_gradcam_for_prediction(
                        legacy_pipeline.image_classifier.model,
                        temp_path,
                        preprocessed,
                        cnn_class_idx,
                    )
                    if overlay_img is not None:
                        gradcam_overlay_b64 = encode_image_bytes(overlay_img, fmt="PNG")
                        gradcam_focus_ratio = compute_gradcam_focus_ratio(heatmap_img)
                except Exception:  # noqa: BLE001 - Grad-CAM is best-effort
                    gradcam_overlay_b64 = None
                    gradcam_focus_ratio = 0.0

                # Choose the top-1 detection for the legacy single-card fields
                top_detection = max(inference.detections, key=lambda d: d.confidence, default=None)
                if top_detection is None:
                    # ------------------------------------------------------------------
                    # YOLO returned no detections above the confidence threshold
                    # (default 0.45). The legacy CNN was already run above (the
                    # Grad-CAM heatmap was computed there). Use its result for
                    # every page field. This branch keeps the page coherent when
                    # YOLO abstains.
                    # ------------------------------------------------------------------
                    preview = encode_image_preview(temp_path)

                    class_explanation = (
                        "The YOLOv8 multi-object detector found no confident "
                        "detection above the 0.45 threshold, so the legacy CNN "
                        "baseline classifier was used as a fallback. The CNN "
                        f"predicted {legacy_result.predicted_class} with a "
                        f"softmax confidence of {legacy_result.class_confidence*100:.1f} percent."
                    )
                    top_k = _top_k_probabilities(legacy_result.class_probabilities, k=6)
                    confidence_label = get_confidence_label(legacy_result.class_confidence)
                    rdf_explanation_text = build_rdf_explanation(
                        legacy_result.predicted_class,
                        legacy_result.material_features,
                        legacy_result.rdf_label,
                        legacy_result.rdf_probability,
                    )
                    environmental_explanation = build_environmental_explanation(
                        legacy_result.predicted_class,
                    )
                    waste_info = get_waste_info(legacy_result.predicted_class)
                    visual_traits = get_visual_traits(legacy_result.predicted_class)

                    # Synthesise one "detection" so the legacy result flows through
                    # the same template path. Class is the CNN's top-1; the bbox is
                    # a full-image placeholder (0,0,1,1) because the CNN doesn't
                    # emit boxes. The UI renders this as one detection spanning
                    # the whole image, which is exactly what a single-label
                    # classifier implies.
                    full_image_bbox = (0.0, 0.0, 1.0, 1.0)
                    detections_payload = [{
                        "class_name": legacy_result.predicted_class,
                        "confidence": legacy_result.class_confidence,
                        "bbox_xyxy": full_image_bbox,
                        "material_features": legacy_result.material_features,
                        "rdf_suitability": legacy_result.rdf_suitability,
                        "rdf_probability": legacy_result.rdf_probability,
                        "rdf_label": legacy_result.rdf_label,
                    }]

                    result = {
                        "predicted_class": legacy_result.predicted_class,
                        "class_confidence": legacy_result.class_confidence,
                        "rdf_label": legacy_result.rdf_label,
                        "rdf_probability": legacy_result.rdf_probability,
                        "material_features": legacy_result.material_features,
                        "top_k_probabilities": top_k,
                        "confidence_label": confidence_label,
                        "class_explanation": class_explanation,
                        "rdf_explanation_text": rdf_explanation_text,
                        "environmental_explanation": environmental_explanation,
                        "visual_traits": visual_traits,
                        "waste_info": waste_info,
                        "gradcam_overlay": gradcam_overlay_b64,
                        "gradcam_focus_ratio": gradcam_focus_ratio,
                        "detections": detections_payload,
                        "num_objects": 1,
                        "image_width": 0,
                        "image_height": 0,
                    }
                    # Skip the rest of the YOLO-only block by jumping to render
                    return render_template(
                        "index.html",
                        result=result,
                        preview=preview,
                        error=None,
                        limitations=LIMITATIONS,
                    )

                # The legacy CNN was already run above; gradcam_overlay_b64,
                # gradcam_focus_ratio and legacy_result are already set
                # from that pre-flight call. The YOLO detection is used for
                # the bounding-box card; the CNN's heatmap lights up
                # independently of YOLO's confidence.

                # Encode the original preview
                preview = encode_image_preview(temp_path)

                # Build the object-level explanation data
                # When YOLO and the CNN disagree on the top-1 class, surface
                # both predictions explicitly. Grad-CAM visualises what the
                # CNN attended to, which is the second opinion.
                cnn_top_class = legacy_result.predicted_class
                cnn_top_conf = legacy_result.class_confidence
                if cnn_top_class.lower() != top_detection.class_name.lower():
                    class_explanation = (
                        f"The YOLOv8 detector identified {inference.num_objects} waste object(s) "
                        f"in this image, with the most confident detection being "
                        f"{top_detection.class_name} ({top_detection.confidence*100:.1f}%). "
                        f"The legacy CNN baseline classifier independently predicted "
                        f"{cnn_top_class} ({cnn_top_conf*100:.1f}%); the Grad-CAM panel "
                        f"below shows which pixels the CNN attended to. When the two "
                        f"models disagree, the Random Forest still uses YOLO's "
                        f"detection for the per-object RDF verdict because the "
                        f"Random Forest expects multi-object grounding."
                    )
                else:
                    class_explanation = (
                        f"The YOLOv8 detector identified {inference.num_objects} waste object(s) "
                        f"in this image. The most confident detection is "
                        f"{top_detection.class_name} with a confidence of "
                        f"{top_detection.confidence*100:.1f} percent. The legacy CNN "
                        f"agrees ({cnn_top_conf*100:.1f}%), and Grad-CAM visualises the "
                        f"CNN's focus region."
                    )
                probabilities = {
                    det.class_name: det.confidence for det in inference.detections
                }
                top_k = _top_k_probabilities(probabilities, k=6)
                confidence_label = get_confidence_label(top_detection.confidence)
                rdf_explanation_text = build_rdf_explanation(
                    top_detection.class_name,
                    top_detection.material_features,
                    top_detection.rdf_label,
                    top_detection.rdf_probability,
                )
                environmental_explanation = build_environmental_explanation(
                    top_detection.class_name
                )
                waste_info = get_waste_info(top_detection.class_name)
                visual_traits = get_visual_traits(top_detection.class_name)

                detections_payload = [
                    {
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "bbox_xyxy": det.bbox_xyxy,
                        "material_features": det.material_features,
                        "rdf_suitability": det.rdf_suitability,
                        "rdf_probability": det.rdf_probability,
                        "rdf_label": det.rdf_label,
                    }
                    for det in inference.detections
                ]

                result = {
                    # --- legacy single-label fields (still bound for template compat) ---
                    "predicted_class": top_detection.class_name,
                    "class_confidence": top_detection.confidence,
                    "rdf_label": top_detection.rdf_label,
                    "rdf_probability": top_detection.rdf_probability,
                    "material_features": top_detection.material_features,
                    # --- new presentation-only fields ---
                    "top_k_probabilities": top_k,
                    "confidence_label": confidence_label,
                    "class_explanation": class_explanation,
                    "rdf_explanation_text": rdf_explanation_text,
                    "environmental_explanation": environmental_explanation,
                    "visual_traits": visual_traits,
                    "waste_info": waste_info,
                    "gradcam_overlay": gradcam_overlay_b64,
                    "gradcam_focus_ratio": gradcam_focus_ratio,
                    # --- multi-object detection payload ---
                    "detections": detections_payload,
                    "num_objects": inference.num_objects,
                    "image_width": inference.image_width,
                    "image_height": inference.image_height,
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
        limitations=LIMITATIONS,
    )


if __name__ == "__main__":
    app.run(debug=True)
