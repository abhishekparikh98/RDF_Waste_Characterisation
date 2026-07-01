"""
Configuration schema and default settings for CNN baseline model
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    
    img_height: int = 224
    img_width: int = 224
    batch_size: int = 32
    num_classes: int = 6
    class_names: list = None
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


@dataclass
class ModelConfig:
    """CNN model architecture configuration"""
    
    num_classes: int = 6
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    dropout_rate: float = 0.5
    conv_filters_block1: int = 32
    conv_filters_block2: int = 64
    conv_filters_block3: int = 128
    dense_units: int = 256


@dataclass
class TrainingConfig:
    """Training process configuration"""
    
    learning_rate: float = 0.001
    epochs: int = 30
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    validation_split: float = 0.0  # Use separate validation set
    random_seed: int = 42
    
    # Optimizer
    optimizer: str = "adam"
    loss_fn: str = "categorical_crossentropy"
    metrics: list = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy']


@dataclass
class ExperimentConfig:
    """Experiment-level configuration"""
    
    experiment_name: str = "cnn_baseline"
    project_name: str = "trash_classification"
    seed: int = 42
    verbose: int = 1
    
    # Paths
    model_save_path: str = "models/cnn_baseline_best.h5"
    results_dir: str = "results/"
    reports_dir: str = "reports/"
    log_file: str = "training.log"


# Default configurations
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
