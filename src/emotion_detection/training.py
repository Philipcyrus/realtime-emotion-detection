"""Keras training pipeline for emotion classification models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from emotion_detection.dataset import discover_labels, load_image_folder_arrays, parse_labels


def build_emotion_model(num_classes: int, image_size: int = 48) -> Any:
    """Build an augmented CNN suitable for FER-style grayscale emotion inputs."""

    import tensorflow as tf

    regularizer = tf.keras.regularizers.l2(1e-4)

    def conv_block(filters: int, dropout: float) -> list[Any]:
        return [
            tf.keras.layers.Conv2D(
                filters,
                3,
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizer,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv2D(
                filters,
                3,
                padding="same",
                use_bias=False,
                kernel_regularizer=regularizer,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(dropout),
        ]

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 1)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            *conv_block(32, 0.10),
            *conv_block(64, 0.15),
            *conv_block(128, 0.25),
            *conv_block(256, 0.35),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.45),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def train_model(
    train_dir: Path,
    validation_dir: Path,
    model_path: Path,
    metadata_path: Path | None = None,
    labels: str | None = None,
    epochs: int = 8,
    batch_size: int = 32,
    image_size: int = 48,
    learning_rate: float = 3e-4,
    patience: int = 8,
) -> dict[str, Any]:
    """Train and save a Keras emotion model from image-folder datasets."""

    import tensorflow as tf

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not validation_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {validation_dir}")

    class_names = parse_labels(labels) if labels else discover_labels(train_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_path or model_path.with_suffix(".metadata.json")

    train_x, train_y = load_image_folder_arrays(train_dir, class_names, image_size)
    validation_x, validation_y = load_image_folder_arrays(
        validation_dir,
        class_names,
        image_size,
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_x, train_y))
        .shuffle(buffer_size=len(train_x), seed=42)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_ds = (
        tf.data.Dataset.from_tensor_slices((validation_x, validation_y))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    class_weights = _class_weights(train_y)

    model = build_emotion_model(num_classes=len(class_names), image_size=image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 3),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    model.save(model_path)

    metrics = {
        key: float(values[-1]) for key, values in history.history.items() if values
    }
    best_metrics = _best_metrics(history.history)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "architecture": "augmented_cnn_v2",
        "labels": list(class_names),
        "input_shape": [image_size, image_size, 1],
        "train_data": str(train_dir),
        "validation_data": str(validation_dir),
        "epochs": epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "train_samples": int(len(train_x)),
        "validation_samples": int(len(validation_x)),
        "class_counts": _class_counts(train_y, class_names),
        "class_weights": {str(key): float(value) for key, value in class_weights.items()},
        "model_path": str(model_path),
        "metrics": metrics,
        "best_metrics": best_metrics,
        "note": "FER2013 labels are noisy; expect improvement but not perfect webcam emotion recognition.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _class_weights(targets: np.ndarray) -> dict[int, float]:
    class_indices = np.argmax(targets, axis=1)
    counts = np.bincount(class_indices, minlength=targets.shape[1])
    total = len(class_indices)
    weights: dict[int, float] = {}
    for index, count in enumerate(counts):
        weights[index] = float(total / (len(counts) * count)) if count else 1.0
    return weights


def _class_counts(targets: np.ndarray, labels: tuple[str, ...]) -> dict[str, int]:
    class_indices = np.argmax(targets, axis=1)
    counts = np.bincount(class_indices, minlength=len(labels))
    return {label: int(counts[index]) for index, label in enumerate(labels)}


def _best_metrics(history: dict[str, list[float]]) -> dict[str, float | int]:
    val_accuracy = history.get("val_accuracy", [])
    if not val_accuracy:
        return {}

    best_index = int(np.argmax(val_accuracy))
    best: dict[str, float | int] = {"best_epoch": best_index + 1}
    for key, values in history.items():
        if values and best_index < len(values):
            best[key] = float(values[best_index])
    return best
