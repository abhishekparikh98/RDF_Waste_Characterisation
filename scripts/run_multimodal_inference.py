"""
Run multimodal inference from the command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_inference import (
    ImageModelConfig,
    MultimodalInferencePipeline,
    RDFModelConfig,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run image-to-RDF multimodal inference")
    parser.add_argument("--image", required=True, help="Path to an input waste image")
    parser.add_argument(
        "--image-model",
        default=str(project_root / "models" / "cnn_baseline_best.h5"),
        help="Path to the trained image classifier model",
    )
    parser.add_argument(
        "--rdf-model",
        default=str(project_root / "models" / "rdf_random_forest_pipeline.joblib"),
        help="Path to the trained RDF suitability model",
    )
    parser.add_argument(
        "--preprocess-mode",
        default="baseline",
        choices=["baseline", "mobilenetv2", "resnet50"],
        help="Image preprocessing mode matching the trained classifier",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the inference result as JSON",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the multimodal inference pipeline."""
    args = parse_args()

    pipeline = MultimodalInferencePipeline(
        image_model_config=ImageModelConfig(
            model_path=Path(args.image_model),
            preprocess_mode=args.preprocess_mode,
        ),
        rdf_model_config=RDFModelConfig(model_path=Path(args.rdf_model)),
    )

    result = pipeline.infer(Path(args.image))
    payload = {
        "image_path": result.image_path,
        "predicted_class": result.predicted_class,
        "class_confidence": result.class_confidence,
        "material_features": result.material_features,
        "rdf_suitability": result.rdf_suitability,
        "rdf_probability": result.rdf_probability,
        "rdf_label": result.rdf_label,
    }

    print(json.dumps(payload, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
