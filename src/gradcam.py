"""
Grad-CAM implementation for Keras image classifiers.

Produces a class-activation heatmap that highlights the image regions
responsible for the model's prediction. No retraining is required.

The module is model-agnostic: it auto-detects the last 4D convolutional
layer in any Keras model. If the loaded model has no 4D conv layer (e.g.
a pure MLP), Grad-CAM returns ``None`` and the UI falls back gracefully.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", 2017.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _find_last_conv_layer(model: keras.Model) -> Optional[keras.layers.Layer]:
    """Return the last 4D-output Conv2D layer, or None if none exists.

    Works with both built and unbuilt models: if a Sequential model
    has not been built yet, the per-layer ``output_shape`` attribute
    is not populated. In that case, we fall back to the layer's
    ``__class__.__name__`` check and pick the last Conv2D.
    """
    # Try the strict check first (works on built models).
    for layer in reversed(model.layers):
        shape = getattr(layer, "output_shape", None)
        if isinstance(shape, tuple) and len(shape) == 4 and isinstance(layer, keras.layers.Conv2D):
            return layer
    # Fall back: pick the last Conv2D by class name.
    last_conv = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = layer
    return last_conv


def _ensure_model_built(model: keras.Model, input_shape: Tuple[int, int, int]) -> None:
    """If the model hasn't been built yet, build it with a known shape.

    Some Sequential models loaded from .h5 do not have an output
    graph attached until the first forward pass. Calling
    ``model.build(...)`` once attaches the layer graph and lets us
    access intermediate activations by name.
    """
    if hasattr(model, "_is_graph_network") and model._is_graph_network:
        return
    # Sequential with no graph yet: build it.
    try:
        if not getattr(model, "built", False):
            model.build((None,) + tuple(input_shape))
    except Exception:  # noqa: BLE001 - building is best-effort
        pass


def _build_heatmap_model(
    model: keras.Model, last_conv_layer: keras.layers.Conv2D
) -> keras.Model:
    """Construct a sub-model that outputs the conv activations and the final prediction.

    The sub-model takes the same input as ``model`` and produces two
    outputs: the activations of ``last_conv_layer`` and the final
    prediction tensor.
    """
    # Walk the model layers in order from the input. For Sequential
    # models that have not been called yet, we cannot read
    # ``model.input`` directly. Build the gradient graph manually
    # by chaining the layers ourselves.
    layers_in_order = list(model.layers)
    # Find the index of the last conv layer and the output
    last_conv_index = None
    for i, layer in enumerate(layers_in_order):
        if layer is last_conv_layer:
            last_conv_index = i
            break
    if last_conv_index is None:
        raise ValueError("Last conv layer not found in model.layers")

    # Build a fresh input tensor with the model's known input shape.
    input_shape = model.input_shape if hasattr(model, "input_shape") else (None, 224, 224, 3)
    input_tensor = keras.Input(shape=input_shape[1:])

    # Forward pass through every layer up to and including the last conv
    x = input_tensor
    for layer in layers_in_order[: last_conv_index + 1]:
        x = layer(x)
    conv_output = x

    # Forward pass through the rest of the network
    y = x
    for layer in layers_in_order[last_conv_index + 1 :]:
        y = layer(y)
    final_output = y

    return keras.Model(inputs=input_tensor, outputs=[conv_output, final_output])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_gradcam_heatmap(
    model: keras.Model,
    image_array: np.ndarray,
    class_index: int,
) -> Optional[np.ndarray]:
    """Compute the Grad-CAM heatmap for a single image.

    Args:
        model: A loaded Keras classification model.
        image_array: Preprocessed input batch of shape (1, H, W, 3), float32.
        class_index: Index of the class whose activation is to be explained.

    Returns:
        A 2D numpy array of shape (h, w) with values in [0, 1], or
        ``None`` if the model has no 4D conv layer.
    """
    # Ensure the model has been built so its layer graph is attached.
    _ensure_model_built(model, (image_array.shape[1], image_array.shape[2], image_array.shape[3]))

    last_conv_layer = _find_last_conv_layer(model)
    if last_conv_layer is None:
        return None

    grad_model = _build_heatmap_model(model, last_conv_layer)
    image_array = tf.cast(image_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        # Handle models whose output is a list vs a tensor
        if isinstance(predictions, list):
            predictions = predictions[0]
        # Guard against class_index being out of range
        class_index = int(max(0, min(class_index, int(predictions.shape[-1]) - 1)))
        loss = predictions[:, class_index]

    # Gradients of the target class w.r.t. the conv feature map
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    # Global-average-pool the gradients to get neuron importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of the conv activations using the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU keeps only the positive influence on the target class
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap_on_image(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: Optional[int] = None,
) -> Image.Image:
    """Overlay the heatmap on the original image and return a new PIL Image.

    The heatmap is resized to the original image size, converted to a
    colour map, and blended with the original using ``alpha``.

    Args:
        original_image: The PIL image that was classified.
        heatmap: 2D array of values in [0, 1].
        alpha: Blending weight for the heatmap colour.
        colormap: OpenCV colour-map code. Default is COLORMAP_JET.

    Returns:
        A new PIL.Image of the same size as ``original_image``.
    """
    import cv2  # local import: only needed when Grad-CAM is actually used

    h, w = original_image.size[1], original_image.size[0]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    coloured = cv2.applyColorMap(heatmap_uint8, colormap)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)

    original_rgb = np.asarray(original_image.convert("RGB"), dtype=np.uint8)
    if colormap is None:
        colormap = cv2_colormap_jets()
    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, coloured, alpha, 0)
    return Image.fromarray(overlay)


def cv2_colormap_jets() -> int:
    """Return OpenCV's COLORMAP_JET constant. Defined as a function so
    the module can be imported even when opencv-python is unavailable.
    """
    import cv2
    return cv2.COLORMAP_JET


def generate_gradcam_for_prediction(
    model: keras.Model,
    image_path: str | Path,
    preprocessed_batch: np.ndarray,
    class_index: int,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """Convenience wrapper used by the Flask app.

    Args:
        model: Loaded Keras model.
        image_path: Path to the original image on disk.
        preprocessed_batch: The same batch passed to ``model.predict``.
        class_index: Index of the predicted class.

    Returns:
        A tuple (heatmap_image, overlay_image). Either element may be
        ``None`` if Grad-CAM could not be produced (e.g. model has no
        conv layer, or cv2 is unavailable).
    """
    try:
        heatmap = compute_gradcam_heatmap(model, preprocessed_batch, class_index)
    except Exception:  # noqa: BLE001 - Grad-CAM is best-effort
        return None, None
    if heatmap is None:
        return None, None

    try:
        original = Image.open(image_path).convert("RGB")
    except Exception:  # noqa: BLE001
        return None, None

    try:
        overlay = overlay_heatmap_on_image(original, heatmap)
    except Exception:  # noqa: BLE001 - cv2 may be missing
        return None, None

    # Also return the raw heatmap as a colour-mapped PIL image so the UI
    # can show the heatmap on its own if desired.
    try:
        import cv2
        h, w = original.size[1], original.size[0]
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
        heatmap_image = Image.fromarray(coloured)
    except Exception:  # noqa: BLE001
        heatmap_image = None

    return heatmap_image, overlay
