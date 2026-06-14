"""Still-image inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from emotion_detection.emotion_model import KerasEmotionClassifier
from emotion_detection.face_detection import HaarFaceDetector


def predict_image(
    image_path: Path,
    model_path: Path,
    labels: tuple[str, ...],
    allow_full_image: bool = True,
) -> dict[str, Any]:
    """Predict emotion for the first detected face in an image."""

    import cv2

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    classifier = KerasEmotionClassifier(model_path, labels)
    detector = HaarFaceDetector()
    faces = detector.detect(image)

    if faces:
        face = faces[0]
        crop = image[face.y:face.bottom, face.x:face.right]
        prediction = classifier.predict(crop)
        face_payload: dict[str, int] | None = {
            "x": face.x,
            "y": face.y,
            "width": face.width,
            "height": face.height,
        }
        mode = "detected_face"
    elif allow_full_image:
        prediction = classifier.predict(image)
        face_payload = None
        mode = "full_image_fallback"
    else:
        raise RuntimeError("No face detected in image.")

    return {
        "image_path": str(image_path),
        "model_path": str(model_path),
        "mode": mode,
        "face": face_payload,
        "prediction": {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "source": prediction.source,
        },
    }
