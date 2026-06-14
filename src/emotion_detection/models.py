"""Shared data models for detection results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceBox:
    """Pixel coordinates for a detected face."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class EmotionPrediction:
    """Emotion label and score for a single face."""

    label: str
    confidence: float
    source: str


@dataclass(frozen=True)
class DetectionResult:
    """A detected face paired with an emotion prediction."""

    face: FaceBox
    prediction: EmotionPrediction
