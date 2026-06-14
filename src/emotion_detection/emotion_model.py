"""Emotion classifier implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from emotion_detection.models import EmotionPrediction
from emotion_detection.preprocessing import preprocess_face


class EmotionClassifier(Protocol):
    """Classifier interface used by the webcam loop."""

    source: str
    warning: str | None

    def predict(self, face_bgr: Any) -> EmotionPrediction:
        """Predict the most likely emotion for a BGR face crop."""


class KerasEmotionClassifier:
    """TensorFlow/Keras-backed emotion classifier."""

    source = "keras"

    def __init__(self, model_path: Path, labels: tuple[str, ...]) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        from tensorflow.keras.models import load_model

        self._model = load_model(model_path)
        self._labels = labels
        self.warning = _metadata_warning(model_path)

    def predict(self, face_bgr: Any) -> EmotionPrediction:
        batch = preprocess_face(face_bgr)
        probabilities = np.asarray(self._model.predict(batch, verbose=0))[0]
        best_index = int(np.argmax(probabilities))
        label = self._labels[best_index] if best_index < len(self._labels) else str(best_index)
        confidence = float(probabilities[best_index])
        return EmotionPrediction(label=label, confidence=confidence, source=self.source)


class DemoEmotionClassifier:
    """A deterministic placeholder that keeps the webcam demo usable.

    This is not a scientific emotion model. It exists so the application can be
    explored before a trained Keras asset is added.
    """

    source = "demo"
    warning = "Demo classifier is heuristic and not a trained emotion model."

    def __init__(self, labels: tuple[str, ...]) -> None:
        self._labels = labels

    def predict(self, face_bgr: Any) -> EmotionPrediction:
        import cv2

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        label = self._choose_label(brightness, contrast)
        confidence = min(0.95, max(0.35, (contrast / 100.0) + 0.35))
        return EmotionPrediction(label=label, confidence=confidence, source=self.source)

    def _choose_label(self, brightness: float, contrast: float) -> str:
        preferred = "neutral"
        if brightness > 150 and "happy" in self._labels:
            preferred = "happy"
        elif brightness < 75 and "sad" in self._labels:
            preferred = "sad"
        elif contrast > 70 and "surprise" in self._labels:
            preferred = "surprise"

        return preferred if preferred in self._labels else self._labels[0]


def create_classifier(
    model_path: Path | None,
    labels: tuple[str, ...],
) -> EmotionClassifier:
    """Create the best available classifier for the provided configuration."""

    if model_path is None:
        print("No model path provided; using demo classifier.")
        return DemoEmotionClassifier(labels)

    try:
        resolved_labels = labels_from_metadata(model_path) or labels
        classifier = KerasEmotionClassifier(model_path, resolved_labels)
        if classifier.warning:
            print(f"Model warning: {classifier.warning}")
        return classifier
    except Exception as exc:
        print(f"Could not load Keras model ({exc}); using demo classifier.")
        return DemoEmotionClassifier(labels)


def labels_from_metadata(model_path: Path) -> tuple[str, ...] | None:
    """Read label order from the sibling metadata file when it exists."""

    metadata = _read_metadata(model_path)
    labels = metadata.get("labels") if metadata else None
    if not labels:
        return None
    return tuple(str(label) for label in labels)


def _metadata_warning(model_path: Path) -> str | None:
    metadata = _read_metadata(model_path)
    if not metadata:
        return None

    note = str(metadata.get("note", "")).strip()
    train_data = str(metadata.get("train_data", ""))
    validation_data = str(metadata.get("validation_data", ""))
    val_accuracy = metadata.get("metrics", {}).get("val_accuracy")

    warnings: list[str] = []
    if note:
        warnings.append(note)
    if "starter_emotions" in train_data or "starter_emotions" in validation_data:
        warnings.append("This is the generated starter model, so predictions are not reliable.")
    if isinstance(val_accuracy, int | float) and val_accuracy < 0.4:
        warnings.append(f"Validation accuracy is low ({val_accuracy:.2f}).")

    return " ".join(warnings) if warnings else None


def _read_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
