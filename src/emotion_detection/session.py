"""Privacy-safe session metrics for webcam emotion detection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from emotion_detection.models import DetectionResult


@dataclass(frozen=True)
class SessionSummary:
    """Aggregated metrics for a webcam run, without image/frame data."""

    session_name: str
    started_at: str
    ended_at: str
    duration_seconds: float
    frames_processed: int
    faces_detected: int
    average_fps: float
    average_confidence: float
    emotion_distribution: dict[str, int]
    timeline: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class SessionTracker:
    """Collect metrics during a live webcam session."""

    def __init__(self, session_name: str) -> None:
        self._session_name = session_name
        self._started = datetime.now(UTC)
        self._frames_processed = 0
        self._fps_values: list[float] = []
        self._confidence_values: list[float] = []
        self._distribution: Counter[str] = Counter()
        self._timeline: list[dict[str, object]] = []

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    def record_frame(self, fps: float) -> None:
        self._frames_processed += 1
        self._fps_values.append(float(fps))

    def record_detection(self, result: DetectionResult) -> None:
        prediction = result.prediction
        face = result.face
        self._distribution[prediction.label] += 1
        self._confidence_values.append(prediction.confidence)
        self._timeline.append(
            {
                "offset_seconds": round(
                    (datetime.now(UTC) - self._started).total_seconds(),
                    3,
                ),
                "label": prediction.label,
                "confidence": round(prediction.confidence, 4),
                "source": prediction.source,
                "face": {
                    "x": face.x,
                    "y": face.y,
                    "width": face.width,
                    "height": face.height,
                },
            }
        )

    def summarize(self) -> SessionSummary:
        ended = datetime.now(UTC)
        duration = max((ended - self._started).total_seconds(), 0.0)
        average_fps = _mean(self._fps_values)
        average_confidence = _mean(self._confidence_values)
        return SessionSummary(
            session_name=self._session_name,
            started_at=self._started.isoformat(),
            ended_at=ended.isoformat(),
            duration_seconds=round(duration, 3),
            frames_processed=self._frames_processed,
            faces_detected=sum(self._distribution.values()),
            average_fps=round(average_fps, 3),
            average_confidence=round(average_confidence, 4),
            emotion_distribution=dict(sorted(self._distribution.items())),
            timeline=self._timeline,
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
