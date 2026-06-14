"""OpenCV face detection utilities."""

from __future__ import annotations

from typing import Any

from emotion_detection.models import FaceBox


class HaarFaceDetector:
    """Frontal face detector using OpenCV's bundled Haar cascade."""

    def __init__(self) -> None:
        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load OpenCV cascade: {cascade_path}")

    def detect(self, frame_bgr: Any) -> list[FaceBox]:
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )
        return [FaceBox(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
