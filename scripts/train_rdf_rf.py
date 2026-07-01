"""
Train a Random Forest model for RDF suitability prediction.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import ConfusionMatrixVisualizer, MetricsCalculator, TabularModelVisualizer
from src.rdf_preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RDFDataConfig,
    RDFPreprocessingPipeline,
)
from src.training import train_rdf_random_forest


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure console and file logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger("rdf_random_forest")
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
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

    return logger


logger = setup_logging(log_file=str(project_root / "rdf_training.log"))


def ensure_directories() -> Dict[str, Path]:
    """Create output directories used by the pipeline."""
    results_dir = project_root / "results"
    reports_dir = project_root / "reports"
    models_dir = project_root / "models"
    data_dir = project_root / "data" / "rdf_features"

    for directory in [results_dir, reports_dir, models_dir, data_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "results": results_dir,
        "reports": reports_dir,
        "models": models_dir,
        "data": data_dir,
    }


def build_class_names() -> List[str]:
    """Return binary class names for RDF suitability."""
    return ["Not Suitable", "Suitable"]


def evaluate_model(
    model,
    X_test,
    y_test,
    class_names: List[str],
    results_dir: Path,
) -> Dict[str, object]:
    """Evaluate the trained model and save metrics artifacts."""
    y_pred = model.predict(X_test)

    metrics_calc = MetricsCalculator(class_names=class_names, average_type="weighted")
    metrics = metrics_calc.calculate_metrics(y_test.to_numpy(), y_pred)
    cm = metrics_calc.get_confusion_matrix(y_test.to_numpy(), y_pred)
    report_text = metrics_calc.get_classification_report(y_test.to_numpy(), y_pred, output_dict=False)

    cm_path = results_dir / "rdf_confusion_matrix.png"
    ConfusionMatrixVisualizer.plot(
        cm=cm,
        class_names=class_names,
        save_path=str(cm_path),
        dpi=300,
    )

    report_path = results_dir / "rdf_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report_text,
        "report_path": report_path,
        "confusion_matrix_path": cm_path,
        "y_pred": y_pred,
    }


def generate_report(
    *,
    dataset_summary: Dict[str, object],
    best_params: Dict[str, object],
    cv_score: float,
    metrics: Dict[str, float],
    report_text: str,
    feature_importance_path: Path,
    confusion_matrix_path: Path,
    model_path: Path,
    results_dir: Path,
    reports_dir: Path,
) -> Path:
    """Generate a markdown report for the Random Forest experiment."""
    report_path = reports_dir / "rdf_random_forest_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Random Forest RDF Suitability Report

**Project:** Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production  
**Generated:** {timestamp}

---

## Dataset Summary

- Samples: {dataset_summary["samples"]}
- Class balance: {dataset_summary["class_balance"]}
- Features: {", ".join(FEATURE_COLUMNS)}
- Target: `{TARGET_COLUMN}`

## Preprocessing

The tabular pipeline applies:
- median imputation for numeric features
- most-frequent imputation for categorical features
- one-hot encoding for `material_type`
- standard scaling for numeric columns

## Model Configuration

- Estimator: Random Forest Classifier
- Best parameters: {json.dumps(best_params, indent=2)}
- Cross-validation score: {cv_score:.4f}

## Test Metrics

| Metric | Value |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1-score | {metrics["f1_score"]:.4f} |

## Classification Report

```text
{report_text}
```

## Artifacts

- Model pipeline: `{model_path}`
- Confusion matrix: `{confusion_matrix_path}`
- Feature importance: `{feature_importance_path}`
- Results directory: `{results_dir}`

## Observations

- Material type is expected to be one of the strongest predictors because it encodes RDF-relevant composition.
- Moisture and contamination should penalize suitability because they lower fuel quality.
- Combustibility and calorific value should contribute positively to the suitability score.

## Limitations

- The dataset is synthetic and based on domain-informed rules, not real plant measurements.
- The model is intentionally isolated from the image classifier for now.
- Future work should validate the feature engineering on industrial RDF records.
"""

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    return report_path


def main() -> None:
    """Run the Random Forest RDF training pipeline."""
    logger.info("=" * 80)
    logger.info("Random Forest RDF Suitability Prediction")
    logger.info("=" * 80)

    dirs = ensure_directories()
    pipeline = RDFPreprocessingPipeline(RDFDataConfig(csv_path=dirs["data"] / "rdf_dataset.csv"))
    df = pipeline.load_or_generate_dataset()

    class_balance = df["rdf_suitable"].value_counts().sort_index().to_dict()
    dataset_summary = {
        "samples": len(df),
        "class_balance": class_balance,
    }

    X_train, X_test, y_train, y_test = pipeline.split_dataset(df)
    preprocessor = pipeline.build_preprocessor()

    search = train_rdf_random_forest(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        cv_folds=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )

    best_model = search.best_estimator_
    model_path = dirs["models"] / "rdf_random_forest_pipeline.joblib"
    joblib.dump(best_model, model_path)

    evaluation = evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        class_names=build_class_names(),
        results_dir=dirs["results"],
    )

    feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
    importances = best_model.named_steps["classifier"].feature_importances_
    feature_importance_path = dirs["results"] / "rdf_feature_importance.png"
    TabularModelVisualizer.plot_feature_importance(
        importances=importances,
        feature_names=list(feature_names),
        title="Random Forest Feature Importance",
        save_path=str(feature_importance_path),
        dpi=300,
    )

    report_path = generate_report(
        dataset_summary=dataset_summary,
        best_params=search.best_params_,
        cv_score=search.best_score_,
        metrics=evaluation["metrics"],
        report_text=evaluation["classification_report"],
        feature_importance_path=feature_importance_path,
        confusion_matrix_path=evaluation["confusion_matrix_path"],
        model_path=model_path,
        results_dir=dirs["results"],
        reports_dir=dirs["reports"],
    )

    logger.info("Model saved to %s", model_path)
    logger.info("Report saved to %s", report_path)
    logger.info("Feature importance saved to %s", feature_importance_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Error during RDF training: %s", exc, exc_info=True)
        raise
