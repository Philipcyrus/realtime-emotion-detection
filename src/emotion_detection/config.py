"""Runtime configuration for the emotion detection app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration values needed by the webcam runtime."""

    camera_index: int = 0
    model_path: Path | None = None
    frame_width: int = 1280
    frame_height: int = 720
    labels: tuple[str, ...] = DEFAULT_LABELS
    confidence_precision: int = 2
    report_dir: Path | None = None
    session_name: str | None = None
    max_frames: int | None = None
    headless: bool = False

    @classmethod
    def from_values(
        cls,
        camera_index: int = 0,
        model: str | None = None,
        width: int = 1280,
        height: int = 720,
        labels: str | None = None,
        report_dir: str | None = None,
        session_name: str | None = None,
        max_frames: int | None = None,
        headless: bool = False,
    ) -> "RuntimeConfig":
        if headless and max_frames is None:
            raise ValueError("Headless mode requires --max-frames so the run can stop.")

        parsed_labels = tuple(
            label.strip() for label in labels.split(",") if label.strip()
        ) if labels else DEFAULT_LABELS

        if not parsed_labels:
            raise ValueError("At least one emotion label is required.")

        return cls(
            camera_index=camera_index,
            model_path=Path(model) if model else None,
            frame_width=width,
            frame_height=height,
            labels=parsed_labels,
            report_dir=Path(report_dir) if report_dir else None,
            session_name=session_name or _default_session_name(),
            max_frames=max_frames,
            headless=headless,
        )


def _default_session_name() -> str:
    return "session-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
