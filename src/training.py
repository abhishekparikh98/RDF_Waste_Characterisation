"""
Training utilities and callbacks for model training
"""

from typing import Tuple, Optional, Dict, Any
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.models import build_rdf_random_forest


logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages model training with callbacks and monitoring"""
    
    def __init__(
        self,
        model_save_path: str = "models/cnn_baseline_best.h5",
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.001,
        verbose: int = 1
    ):
        """
        Initialize training manager.
        
        Args:
            model_save_path: Path to save best model
            early_stopping_patience: Epochs to wait before stopping if no improvement
            early_stopping_min_delta: Minimum change to qualify as improvement
            verbose: Verbosity level for logging
        """
        self.model_save_path = model_save_path
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.verbose = verbose
    
    def get_callbacks(self) -> list:
        """
        Get list of training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_min_delta,
                restore_best_weights=True,
                verbose=self.verbose,
                mode='min'
            ),
            ModelCheckpoint(
                filepath=self.model_save_path,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=self.verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=self.verbose,
                mode='min'
            )
        ]
        return callbacks
    
    def compile_model(
        self,
        model: tf.keras.Model,
        optimizer: str = 'adam',
        learning_rate: float = 0.001,
        loss_fn: str = 'categorical_crossentropy',
        metrics: list = None
    ) -> tf.keras.Model:
        """
        Compile model with specified optimizer and loss function.
        
        Args:
            model: Keras model to compile
            optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate for optimizer
            loss_fn: Loss function name
            metrics: List of metrics to track
        
        Returns:
            Compiled model
        """
        if metrics is None:
            metrics = ['accuracy']
        
        # Create optimizer instance
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=metrics
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer (lr={learning_rate})")
        return model
    
    def train(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        validation_dataset: tf.data.Dataset,
        epochs: int = 30,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """
        Train model with early stopping and model checkpointing.
        
        Args:
            model: Compiled Keras model
            train_dataset: Training dataset
            validation_dataset: Validation dataset
            epochs: Maximum number of epochs
            verbose: Verbosity level (0, 1, or 2)
        
        Returns:
            Training history object
        """
        callbacks = self.get_callbacks()
        
        logger.info(f"Starting training for max {epochs} epochs...")
        
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("Training completed")
        return history


def create_data_loader(
    image_dir: str,
    batch_size: int = 32,
    shuffle: bool = False
) -> tf.data.Dataset:
    """
    Create TensorFlow data loader from directory of images.
    
    Args:
        image_dir: Directory containing image subdirectories (one per class)
        batch_size: Batch size for loading
        shuffle: Whether to shuffle the dataset
    
    Returns:
        tf.data.Dataset ready for training
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        image_dir,
        image_size=(224, 224),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=shuffle,
        seed=42
    )
    
    logger.info(f"Created data loader from {image_dir}")
    return dataset


def prepare_dataset(
    dataset: tf.data.Dataset,
    prefetch: bool = True,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    preprocess_fn = None,
    normalize: bool = True
) -> tf.data.Dataset:
    """
    Optimize dataset performance for training.
    
    Args:
        dataset: Input dataset
        prefetch: Whether to prefetch data
        num_parallel_calls: Number of parallel processing calls
    
    Returns:
        Optimized dataset
    """
    # Normalize pixel values to [0, 1] (images are in [0, 255] from image_dataset_from_directory)
    def normalize_batch(images, labels):
        if normalize:
            images = images / 255.0
        if preprocess_fn is not None:
            images = preprocess_fn(images)
        return images, labels

    dataset = dataset.map(normalize_batch, num_parallel_calls=num_parallel_calls)
    
    if prefetch:
        dataset = dataset.prefetch(buffer_size=num_parallel_calls)
    
    logger.info("Dataset prepared with preprocessing and prefetching")
    return dataset


def train_rdf_random_forest(
    X_train: Any,
    y_train: Any,
    preprocessor: Any,
    param_grid: Dict[str, list] | None = None,
    cv_folds: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """
    Train a Random Forest classifier for RDF suitability prediction.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        preprocessor: Fitted or unfitted sklearn preprocessing pipeline
        param_grid: Grid search parameter combinations
        cv_folds: Number of cross-validation folds
        scoring: Scoring metric for model selection
        n_jobs: Parallel jobs for grid search
        verbose: Grid search verbosity

    Returns:
        Fitted GridSearchCV instance
    """
    if param_grid is None:
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, 30, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", build_rdf_random_forest()),
        ]
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    search.fit(X_train, y_train)

    logger.info("Random Forest training complete; best score %.4f", search.best_score_)
    return search
