"""
Model definitions and architectures.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import ResNet50
from sklearn.ensemble import RandomForestClassifier


class BaselineCNN(Sequential):
    """
    Simple baseline CNN model for waste image classification.
    
    Architecture:
    - 3 convolutional blocks with ReLU activation
    - MaxPooling after each conv block
    - Dropout for regularization
    - Dense output layer with Softmax
    
    Args:
        input_shape: Tuple of (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        conv_filters_block1: Filters in first conv block
        conv_filters_block2: Filters in second conv block
        conv_filters_block3: Filters in third conv block
        dense_units: Units in dense hidden layer
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 6,
        dropout_rate: float = 0.5,
        conv_filters_block1: int = 32,
        conv_filters_block2: int = 64,
        conv_filters_block3: int = 128,
        dense_units: int = 256,
        name: str = "baseline_cnn"
    ):
        super().__init__(name=name)
        
        # Block 1: Conv -> ReLU -> MaxPool -> Dropout
        self.add(layers.Conv2D(
            filters=conv_filters_block1,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            input_shape=input_shape,
            name='conv_block1_conv'
        ))
        self.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            name='conv_block1_pool'
        ))
        self.add(layers.Dropout(
            rate=dropout_rate,
            name='conv_block1_dropout'
        ))
        
        # Block 2: Conv -> ReLU -> MaxPool -> Dropout
        self.add(layers.Conv2D(
            filters=conv_filters_block2,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name='conv_block2_conv'
        ))
        self.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            name='conv_block2_pool'
        ))
        self.add(layers.Dropout(
            rate=dropout_rate,
            name='conv_block2_dropout'
        ))
        
        # Block 3: Conv -> ReLU -> MaxPool -> Dropout
        self.add(layers.Conv2D(
            filters=conv_filters_block3,
            kernel_size=(3, 3),
            padding='same',
            activation='relu',
            name='conv_block3_conv'
        ))
        self.add(layers.MaxPooling2D(
            pool_size=(2, 2),
            name='conv_block3_pool'
        ))
        self.add(layers.Dropout(
            rate=dropout_rate,
            name='conv_block3_dropout'
        ))
        
        # Flatten and Dense layers
        self.add(layers.Flatten(name='flatten'))
        self.add(layers.Dense(
            units=dense_units,
            activation='relu',
            name='dense_hidden'
        ))
        self.add(layers.Dropout(
            rate=dropout_rate,
            name='dense_dropout'
        ))
        
        # Output layer with Softmax
        self.add(layers.Dense(
            units=num_classes,
            activation='softmax',
            name='output'
        ))
    
    def summary_dict(self) -> dict:
        """Return model summary as dictionary"""
        summary = {
            'name': self.name,
            'layers': len(self.layers),
            'total_params': self.count_params(),
            'trainable_params': sum([tf.size(w).numpy() for w in self.trainable_weights])
        }
        return summary


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 6,
    dropout_rate: float = 0.5,
    conv_filters_block1: int = 32,
    conv_filters_block2: int = 64,
    conv_filters_block3: int = 128,
    dense_units: int = 256
) -> BaselineCNN:
    """
    Factory function to build baseline CNN model.
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        conv_filters_block1: Conv filters in block 1
        conv_filters_block2: Conv filters in block 2
        conv_filters_block3: Conv filters in block 3
        dense_units: Dense layer units
    
    Returns:
        Compiled BaselineCNN model
    """
    model = BaselineCNN(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        conv_filters_block1=conv_filters_block1,
        conv_filters_block2=conv_filters_block2,
        conv_filters_block3=conv_filters_block3,
        dense_units=dense_units
    )
    return model


def build_mobilenetv2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 6,
    dropout_rate: float = 0.5,
    dense_units: int = 256,
    trainable_base: bool = False
) -> keras.Model:
    """
    Build a MobileNetV2 transfer-learning model.

    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout probability for the classifier head
        dense_units: Units in the hidden dense layer
        trainable_base: Whether to keep the ImageNet backbone trainable

    Returns:
        Keras model with a frozen MobileNetV2 backbone by default
    """
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_base

    model = Sequential(name="mobilenetv2_transfer")
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D(name="global_average_pooling"))
    model.add(layers.Dense(dense_units, activation="relu", name="dense_hidden"))
    model.add(layers.Dropout(dropout_rate, name="classifier_dropout"))
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    return model


def build_resnet50(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 6,
    dropout_rate: float = 0.5,
    dense_units: int = 256,
    trainable_base: bool = False
) -> keras.Model:
    """
    Build a ResNet50 transfer-learning model.

    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout probability for the classifier head
        dense_units: Units in the hidden dense layer
        trainable_base: Whether to keep the ImageNet backbone trainable

    Returns:
        Keras model with a frozen ResNet50 backbone by default
    """
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = trainable_base

    model = Sequential(name="resnet50_transfer")
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D(name="global_average_pooling"))
    model.add(layers.Dense(dense_units, activation="relu", name="dense_hidden"))
    model.add(layers.Dropout(dropout_rate, name="classifier_dropout"))
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    return model


def build_rdf_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    class_weight: str | None = "balanced_subsample"
) -> RandomForestClassifier:
    """
    Build a Random Forest classifier for RDF suitability prediction.

    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at a leaf node
        random_state: Reproducibility seed
        class_weight: Class balancing strategy

    Returns:
        Configured RandomForestClassifier instance
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )
