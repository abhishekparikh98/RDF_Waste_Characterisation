"""
Flask web application for waste image classification and RDF suitability prediction.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from flask import Flask, flash, render_template, request
from werkzeug.utils import secure_filename

from src.multimodal_inference import (
    ImageModelConfig,
    MultimodalInferencePipeline,
    RDFModelConfig,
)


PROJECT_ROOT = Path(__file__).parent
UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
ALLOWED_MIME_PREFIXES = {"image/"}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "rdf-waste-analytics")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


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


def encode_image_preview(file_path: Path) -> str:
    """Convert an uploaded file into a base64 preview string."""
    mime_type, _ = mimetypes.guess_type(str(file_path))
    mime_type = mime_type or "image/png"
    with open(file_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


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
                inference = pipeline.infer(temp_path)
                preview = encode_image_preview(temp_path)
                result = {
                    "predicted_class": inference.predicted_class,
                    "class_confidence": inference.class_confidence,
                    "rdf_label": inference.rdf_label,
                    "rdf_probability": inference.rdf_probability,
                }
            except Exception as exc:
                error = f"Prediction failed: {exc}"
            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    return render_template("index.html", result=result, preview=preview, error=error)


if __name__ == "__main__":
    app.run(debug=True)
