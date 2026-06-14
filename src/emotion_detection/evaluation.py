"""Saved model evaluation helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from emotion_detection.dataset import (
    discover_labels,
    load_image_folder_arrays,
    parse_labels,
)


def evaluate_model(
    model_path: Path,
    data_dir: Path,
    output_path: Path | None = None,
    labels: str | None = None,
    metadata_path: Path | None = None,
    batch_size: int = 32,
    image_size: int = 48,
) -> dict[str, Any]:
    """Evaluate a saved Keras model against an image-folder dataset."""

    import tensorflow as tf

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {data_dir}")

    class_names = _resolve_labels(data_dir, labels, metadata_path)
    x, y = load_image_folder_arrays(data_dir, class_names, image_size)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

    model = tf.keras.models.load_model(model_path)
    raw_metrics = model.evaluate(dataset, verbose=0, return_dict=True)
    metrics = {name: float(value) for name, value in raw_metrics.items()}

    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "labels": list(class_names),
        "samples": int(len(x)),
        "metrics": metrics,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def _resolve_labels(
    data_dir: Path,
    labels: str | None,
    metadata_path: Path | None,
) -> tuple[str, ...]:
    if labels:
        return parse_labels(labels)
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_labels = metadata.get("labels")
        if metadata_labels:
            return tuple(str(label) for label in metadata_labels)
    return discover_labels(data_dir)
