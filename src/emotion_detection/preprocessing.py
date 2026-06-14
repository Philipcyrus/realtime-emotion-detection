"""Face preprocessing for Keras emotion classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np


def preprocess_face(face_bgr: Any, target_size: tuple[int, int] = (48, 48)) -> np.ndarray:
    """Convert a BGR face crop into a normalized grayscale model batch."""

    import cv2

    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        raise ValueError("Cannot preprocess an empty face crop.")

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=(0, -1))
