"""
Train and compare the baseline CNN, MobileNetV2, and ResNet50 models.
"""

from __future__ import annotations

import io
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DEFAULT_DATA_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_TRAINING_CONFIG, DEFAULT_EXPERIMENT_CONFIG
from src.evaluation import ConfusionMatrixVisualizer, MetricsCalculator, TrainingHistoryVisualizer
from src.models import build_baseline_cnn, build_mobilenetv2, build_resnet50
from src.training import TrainingManager, prepare_dataset


@dataclass
class ExperimentResult:
    """Container for a model experiment result."""

    name: str
    model: keras.Model
    history: Dict[str, List[float]]
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    y_true: np.ndarray
    y_pred: np.ndarray
    model_path: Path


def setup_logging(log_file: str | None = None) -> logging.Logger:
    """Configure console and file logging."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger = logging.getLogger("cnn_mobilenet_compare")
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


logger = setup_logging(log_file=str(project_root / "comparison.log"))


def load_preprocessed_datasets() -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load train, validation and test datasets from data/processed/."""
    data_dir = project_root / "data" / "processed"
    logger.info("Loading datasets from %s", data_dir)

    train_dataset = keras.preprocessing.image_dataset_from_directory(
        str(data_dir / "train"),
        image_size=(DEFAULT_DATA_CONFIG.img_height, DEFAULT_DATA_CONFIG.img_width),
        batch_size=DEFAULT_DATA_CONFIG.batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=DEFAULT_TRAINING_CONFIG.random_seed,
    )
    val_dataset = keras.preprocessing.image_dataset_from_directory(
        str(data_dir / "validation"),
        image_size=(DEFAULT_DATA_CONFIG.img_height, DEFAULT_DATA_CONFIG.img_width),
        batch_size=DEFAULT_DATA_CONFIG.batch_size,
        label_mode="categorical",
        shuffle=False,
        seed=DEFAULT_TRAINING_CONFIG.random_seed,
    )
    test_dataset = keras.preprocessing.image_dataset_from_directory(
        str(data_dir / "test"),
        image_size=(DEFAULT_DATA_CONFIG.img_height, DEFAULT_DATA_CONFIG.img_width),
        batch_size=DEFAULT_DATA_CONFIG.batch_size,
        label_mode="categorical",
        shuffle=False,
        seed=DEFAULT_TRAINING_CONFIG.random_seed,
    )

    return train_dataset, val_dataset, test_dataset, train_dataset.class_names


def dataset_sample_count(dataset: tf.data.Dataset) -> int:
    """Estimate the number of samples in a batched dataset."""
    cardinality = tf.data.experimental.cardinality(dataset).numpy()
    if cardinality < 0:
        return 0
    return int(cardinality) * DEFAULT_DATA_CONFIG.batch_size


def prepare_for_baseline(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Normalize datasets for the baseline CNN."""
    return (
        prepare_dataset(train_dataset),
        prepare_dataset(val_dataset),
        prepare_dataset(test_dataset),
    )


def prepare_for_mobilenet(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Prepare datasets for MobileNetV2 preprocessing."""
    return (
        prepare_dataset(train_dataset, preprocess_fn=mobilenet_preprocess_input, normalize=False),
        prepare_dataset(val_dataset, preprocess_fn=mobilenet_preprocess_input, normalize=False),
        prepare_dataset(test_dataset, preprocess_fn=mobilenet_preprocess_input, normalize=False),
    )


def prepare_for_resnet(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    test_dataset: tf.data.Dataset,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Prepare datasets for ResNet50 preprocessing."""
    return (
        prepare_dataset(train_dataset, preprocess_fn=resnet_preprocess_input, normalize=False),
        prepare_dataset(val_dataset, preprocess_fn=resnet_preprocess_input, normalize=False),
        prepare_dataset(test_dataset, preprocess_fn=resnet_preprocess_input, normalize=False),
    )


def merge_histories(*histories: keras.callbacks.History) -> Dict[str, List[float]]:
    """Merge multiple training histories into a single history dictionary."""
    merged: Dict[str, List[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, [])
            merged[key].extend(values)
    return merged


def evaluate_model(
    model: keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: List[str],
    results_dir: Path,
    name: str,
) -> Tuple[Dict[str, float], np.ndarray, str, np.ndarray, np.ndarray]:
    """Evaluate a trained model and save the confusion matrix and report."""
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        label_array = labels.numpy()
        if label_array.ndim == 1:
            y_true.extend(label_array.astype(int).tolist())
        else:
            y_true.extend(np.argmax(label_array, axis=1).tolist())

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    metrics_calc = MetricsCalculator(class_names=class_names, average_type="weighted")
    metrics = metrics_calc.calculate_metrics(y_true_np, y_pred_np)
    confusion = metrics_calc.get_confusion_matrix(y_true_np, y_pred_np)
    class_report = metrics_calc.get_classification_report(y_true_np, y_pred_np, output_dict=False)

    ConfusionMatrixVisualizer.plot(
        cm=confusion,
        class_names=class_names,
        save_path=str(results_dir / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"),
        dpi=300,
    )

    report_path = results_dir / f"{name.lower().replace(' ', '_')}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(class_report)

    logger.info("%s metrics: accuracy=%.4f f1=%.4f", name, metrics["accuracy"], metrics["f1_score"])
    return metrics, confusion, class_report, y_true_np, y_pred_np


def train_baseline(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    results_dir: Path,
    models_dir: Path,
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """Train the baseline CNN."""
    model = build_baseline_cnn(
        input_shape=DEFAULT_MODEL_CONFIG.input_shape,
        num_classes=DEFAULT_MODEL_CONFIG.num_classes,
        dropout_rate=DEFAULT_MODEL_CONFIG.dropout_rate,
        conv_filters_block1=DEFAULT_MODEL_CONFIG.conv_filters_block1,
        conv_filters_block2=DEFAULT_MODEL_CONFIG.conv_filters_block2,
        conv_filters_block3=DEFAULT_MODEL_CONFIG.conv_filters_block3,
        dense_units=DEFAULT_MODEL_CONFIG.dense_units,
    )

    trainer = TrainingManager(
        model_save_path=str(models_dir / "cnn_baseline_best.h5"),
        early_stopping_patience=DEFAULT_TRAINING_CONFIG.early_stopping_patience,
        early_stopping_min_delta=DEFAULT_TRAINING_CONFIG.early_stopping_min_delta,
        verbose=1,
    )
    trainer.compile_model(
        model=model,
        optimizer=DEFAULT_TRAINING_CONFIG.optimizer,
        learning_rate=DEFAULT_TRAINING_CONFIG.learning_rate,
        loss_fn=DEFAULT_TRAINING_CONFIG.loss_fn,
        metrics=DEFAULT_TRAINING_CONFIG.metrics,
    )
    history = trainer.train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        epochs=DEFAULT_TRAINING_CONFIG.epochs,
        verbose=1,
    )

    TrainingHistoryVisualizer.plot_training_history(
        history=history.history,
        save_dir=str(results_dir),
        filename_prefix="baseline_",
        dpi=300,
    )

    return model, history.history


def train_mobilenetv2(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    results_dir: Path,
    models_dir: Path,
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """Train MobileNetV2 in two phases: frozen backbone then fine-tuning."""
    model = build_mobilenetv2(
        input_shape=DEFAULT_MODEL_CONFIG.input_shape,
        num_classes=DEFAULT_MODEL_CONFIG.num_classes,
        dropout_rate=DEFAULT_MODEL_CONFIG.dropout_rate,
        dense_units=DEFAULT_MODEL_CONFIG.dense_units,
        trainable_base=False,
    )

    trainer = TrainingManager(
        model_save_path=str(models_dir / "mobilenetv2_best.h5"),
        early_stopping_patience=DEFAULT_TRAINING_CONFIG.early_stopping_patience,
        early_stopping_min_delta=DEFAULT_TRAINING_CONFIG.early_stopping_min_delta,
        verbose=1,
    )

    trainer.compile_model(
        model=model,
        optimizer=DEFAULT_TRAINING_CONFIG.optimizer,
        learning_rate=DEFAULT_TRAINING_CONFIG.learning_rate,
        loss_fn=DEFAULT_TRAINING_CONFIG.loss_fn,
        metrics=DEFAULT_TRAINING_CONFIG.metrics,
    )
    phase_one_history = trainer.train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        epochs=DEFAULT_TRAINING_CONFIG.epochs,
        verbose=1,
    )

    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainer.compile_model(
        model=model,
        optimizer=DEFAULT_TRAINING_CONFIG.optimizer,
        learning_rate=1e-5,
        loss_fn=DEFAULT_TRAINING_CONFIG.loss_fn,
        metrics=DEFAULT_TRAINING_CONFIG.metrics,
    )
    phase_two_history = trainer.train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        epochs=10,
        verbose=1,
    )

    merged_history = merge_histories(phase_one_history, phase_two_history)
    TrainingHistoryVisualizer.plot_training_history(
        history=merged_history,
        save_dir=str(results_dir),
        filename_prefix="mobilenetv2_",
        dpi=300,
    )

    return model, merged_history


def train_resnet50(
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    results_dir: Path,
    models_dir: Path,
) -> Tuple[keras.Model, Dict[str, List[float]]]:
    """Train ResNet50 in two phases: frozen backbone then fine-tuning."""
    model = build_resnet50(
        input_shape=DEFAULT_MODEL_CONFIG.input_shape,
        num_classes=DEFAULT_MODEL_CONFIG.num_classes,
        dropout_rate=DEFAULT_MODEL_CONFIG.dropout_rate,
        dense_units=DEFAULT_MODEL_CONFIG.dense_units,
        trainable_base=False,
    )

    trainer = TrainingManager(
        model_save_path=str(models_dir / "resnet50_best.h5"),
        early_stopping_patience=DEFAULT_TRAINING_CONFIG.early_stopping_patience,
        early_stopping_min_delta=DEFAULT_TRAINING_CONFIG.early_stopping_min_delta,
        verbose=1,
    )

    trainer.compile_model(
        model=model,
        optimizer=DEFAULT_TRAINING_CONFIG.optimizer,
        learning_rate=DEFAULT_TRAINING_CONFIG.learning_rate,
        loss_fn=DEFAULT_TRAINING_CONFIG.loss_fn,
        metrics=DEFAULT_TRAINING_CONFIG.metrics,
    )
    phase_one_history = trainer.train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        epochs=DEFAULT_TRAINING_CONFIG.epochs,
        verbose=1,
    )

    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainer.compile_model(
        model=model,
        optimizer=DEFAULT_TRAINING_CONFIG.optimizer,
        learning_rate=1e-5,
        loss_fn=DEFAULT_TRAINING_CONFIG.loss_fn,
        metrics=DEFAULT_TRAINING_CONFIG.metrics,
    )
    phase_two_history = trainer.train(
        model=model,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        epochs=10,
        verbose=1,
    )

    merged_history = merge_histories(phase_one_history, phase_two_history)
    TrainingHistoryVisualizer.plot_training_history(
        history=merged_history,
        save_dir=str(results_dir),
        filename_prefix="resnet50_",
        dpi=300,
    )

    return model, merged_history


def plot_comparison(results: Dict[str, ExperimentResult], results_dir: Path) -> Path:
    """Create a bar chart comparing the evaluation metrics."""
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    display_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    model_names = list(results.keys())
    x = np.arange(len(metric_names))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    for index, model_name in enumerate(model_names):
        offsets = x + ((index - (len(model_names) - 1) / 2.0) * width)
        values = [results[model_name].metrics[name] for name in metric_names]
        ax.bar(offsets, values, width=width, label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Baseline CNN, MobileNetV2, and ResNet50 Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    output_path = results_dir / "cnn_mobilenetv2_resnet50_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return output_path


def generate_comparison_report(
    results: Dict[str, ExperimentResult],
    class_names: List[str],
    results_dir: Path,
    reports_dir: Path,
    plot_path: Path,
    train_samples: int,
    val_samples: int,
    test_samples: int,
) -> Path:
    """Write a markdown report summarizing the comparison."""
    report_path = reports_dir / "cnn_mobilenetv2_resnet50_evaluation_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    baseline = results["Baseline CNN"]
    mobilenet = results["MobileNetV2"]
    resnet = results["ResNet50"]

    report = f"""# Comparative Evaluation: Baseline CNN, MobileNetV2, and ResNet50

**Project:** Multi-Modal Waste Characterisation for High-Quality Refuse-Derived Fuel Production  
**Dataset:** TrashNet  
**Generated:** {timestamp}

---

## Dataset Summary

- Training samples: {train_samples}
- Validation samples: {val_samples}
- Test samples: {test_samples}
- Classes: {', '.join(class_names)}

---

## Model Comparison

| Metric | Baseline CNN | MobileNetV2 | ResNet50 |
|---|---:|---:|---:|
| Accuracy | {baseline.metrics['accuracy']:.4f} | {mobilenet.metrics['accuracy']:.4f} | {resnet.metrics['accuracy']:.4f} |
| Precision | {baseline.metrics['precision']:.4f} | {mobilenet.metrics['precision']:.4f} | {resnet.metrics['precision']:.4f} |
| Recall | {baseline.metrics['recall']:.4f} | {mobilenet.metrics['recall']:.4f} | {resnet.metrics['recall']:.4f} |
| F1-score | {baseline.metrics['f1_score']:.4f} | {mobilenet.metrics['f1_score']:.4f} | {resnet.metrics['f1_score']:.4f} |

### Relative Change vs Baseline

| Metric | MobileNetV2 vs Baseline | ResNet50 vs Baseline |
|---|---:|---:|
| Accuracy | {mobilenet.metrics['accuracy'] - baseline.metrics['accuracy']:+.4f} | {resnet.metrics['accuracy'] - baseline.metrics['accuracy']:+.4f} |
| Precision | {mobilenet.metrics['precision'] - baseline.metrics['precision']:+.4f} | {resnet.metrics['precision'] - baseline.metrics['precision']:+.4f} |
| Recall | {mobilenet.metrics['recall'] - baseline.metrics['recall']:+.4f} | {resnet.metrics['recall'] - baseline.metrics['recall']:+.4f} |
| F1-score | {mobilenet.metrics['f1_score'] - baseline.metrics['f1_score']:+.4f} | {resnet.metrics['f1_score'] - baseline.metrics['f1_score']:+.4f} |

## Training Setup

### Baseline CNN
- 3 convolutional blocks
- Adam optimizer
- Categorical cross-entropy loss
- Early stopping and model checkpointing

### MobileNetV2
- Frozen ImageNet backbone for the first phase
- Fine-tuning of the top 30 backbone layers
- Adam optimizer with reduced learning rate during fine-tuning
- Same loss and monitoring strategy as baseline

### ResNet50
- Frozen ImageNet backbone for the first phase
- Fine-tuning of the top 30 backbone layers
- Deeper residual representation than MobileNetV2
- Same optimizer, loss, and callback strategy

## Academic Evaluation

### Methodology

All models were trained on the same processed TrashNet split using the same categorical label space and held-out test set. Transfer-learning models used ImageNet initialisation and a two-stage training strategy:
1. Train the classifier head with the backbone frozen.
2. Fine-tune the top backbone layers with a lower learning rate.

### Interpretation

- **Baseline CNN** provides the from-scratch reference point.
- **MobileNetV2** is expected to offer strong efficiency and generalisation for limited data.
- **ResNet50** offers a deeper residual backbone that may improve representation learning at the cost of higher complexity.

### Selected Model

The preferred model should be chosen by test F1-score and validation stability, not training accuracy alone.

## Artifacts

- Baseline model: `{baseline.model_path}`
- MobileNetV2 model: `{mobilenet.model_path}`
- ResNet50 model: `{resnet.model_path}`
- Comparison plot: `{plot_path}`
- Baseline confusion matrix: `results/baseline_cnn_confusion_matrix.png`
- MobileNetV2 confusion matrix: `results/mobilenetv2_confusion_matrix.png`
- ResNet50 confusion matrix: `results/resnet50_confusion_matrix.png`

## Observations

- The baseline CNN is the simpler model and serves as the from-scratch reference.
- MobileNetV2 should typically generalize better because it reuses pretrained visual features.
- ResNet50 may outperform MobileNetV2 when the deeper residual feature hierarchy is beneficial.
- The final comparison should be interpreted on the same held-out test split.

## Notes

The full class-wise classification reports are stored in `results/` for all models.

---

*Generated by Copilot*
"""

    os.makedirs(reports_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    return report_path


def main() -> None:
    """Run the full baseline-vs-MobileNetV2 comparison pipeline."""
    logger.info("=" * 80)
    logger.info("Baseline CNN vs MobileNetV2 Comparison")
    logger.info("=" * 80)

    results_dir = project_root / DEFAULT_EXPERIMENT_CONFIG.results_dir
    reports_dir = project_root / DEFAULT_EXPERIMENT_CONFIG.reports_dir
    models_dir = project_root / "models"
    for directory in [results_dir, reports_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)

    train_dataset, val_dataset, test_dataset, class_names = load_preprocessed_datasets()
    train_samples = dataset_sample_count(train_dataset)
    val_samples = dataset_sample_count(val_dataset)
    test_samples = dataset_sample_count(test_dataset)

    baseline_train, baseline_val, baseline_test = prepare_for_baseline(
        train_dataset, val_dataset, test_dataset
    )
    mobilenet_train, mobilenet_val, mobilenet_test = prepare_for_mobilenet(
        train_dataset, val_dataset, test_dataset
    )
    resnet_train, resnet_val, resnet_test = prepare_for_resnet(
        train_dataset, val_dataset, test_dataset
    )

    baseline_model, baseline_history = train_baseline(
        baseline_train, baseline_val, results_dir, models_dir
    )
    mobilenet_model, mobilenet_history = train_mobilenetv2(
        mobilenet_train, mobilenet_val, results_dir, models_dir
    )
    resnet_model, resnet_history = train_resnet50(
        resnet_train, resnet_val, results_dir, models_dir
    )

    baseline_metrics, baseline_cm, baseline_report, baseline_y_true, baseline_y_pred = evaluate_model(
        baseline_model, baseline_test, class_names, results_dir, "baseline_cnn"
    )
    mobilenet_metrics, mobilenet_cm, mobilenet_report, mobilenet_y_true, mobilenet_y_pred = evaluate_model(
        mobilenet_model, mobilenet_test, class_names, results_dir, "mobilenetv2"
    )
    resnet_metrics, resnet_cm, resnet_report, resnet_y_true, resnet_y_pred = evaluate_model(
        resnet_model, resnet_test, class_names, results_dir, "resnet50"
    )

    results = {
        "Baseline CNN": ExperimentResult(
            name="Baseline CNN",
            model=baseline_model,
            history=baseline_history,
            metrics=baseline_metrics,
            confusion_matrix=baseline_cm,
            classification_report=baseline_report,
            y_true=baseline_y_true,
            y_pred=baseline_y_pred,
            model_path=models_dir / "cnn_baseline_best.h5",
        ),
        "MobileNetV2": ExperimentResult(
            name="MobileNetV2",
            model=mobilenet_model,
            history=mobilenet_history,
            metrics=mobilenet_metrics,
            confusion_matrix=mobilenet_cm,
            classification_report=mobilenet_report,
            y_true=mobilenet_y_true,
            y_pred=mobilenet_y_pred,
            model_path=models_dir / "mobilenetv2_best.h5",
        ),
        "ResNet50": ExperimentResult(
            name="ResNet50",
            model=resnet_model,
            history=resnet_history,
            metrics=resnet_metrics,
            confusion_matrix=resnet_cm,
            classification_report=resnet_report,
            y_true=resnet_y_true,
            y_pred=resnet_y_pred,
            model_path=models_dir / "resnet50_best.h5",
        ),
    }

    plot_path = plot_comparison(results, results_dir)
    report_path = generate_comparison_report(
        results=results,
        class_names=class_names,
        results_dir=results_dir,
        reports_dir=reports_dir,
        plot_path=plot_path,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )

    logger.info("=" * 80)
    logger.info("Comparison complete")
    logger.info("Report saved to %s", report_path)
    logger.info("Comparison plot saved to %s", plot_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Error during comparison run: %s", exc, exc_info=True)
        raise
