"""Environment and project health checks."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path
from typing import Any


def run_doctor(
    model_path: Path | None = None,
    camera_max_index: int = 3,
) -> dict[str, Any]:
    """Return a JSON-serializable health report for the local project."""

    report: dict[str, Any] = {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "imports": {
            "cv2": _import_version("cv2", "__version__"),
            "numpy": _import_version("numpy", "__version__"),
            "tensorflow": _import_version("tensorflow", "__version__"),
            "emotion_detection": _import_version("emotion_detection", "__version__"),
        },
        "model": _model_status(model_path),
        "cameras": _probe_cameras(camera_max_index),
    }
    report["ok"] = all(item["ok"] for item in report["imports"].values())
    return report


def _import_version(module_name: str, version_attr: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "version": str(getattr(module, version_attr, "unknown")),
    }


def _model_status(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {"ok": False, "path": None, "message": "No model path supplied."}
    return {
        "ok": model_path.exists(),
        "path": str(model_path),
        "message": "Model file found." if model_path.exists() else "Model file missing.",
    }


def _probe_cameras(max_index: int) -> list[dict[str, Any]]:
    import cv2

    results = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index)
        opened = capture.isOpened()
        frame_read = False
        frame_shape = None
        if opened:
            frame_read, frame = capture.read()
            frame_shape = list(frame.shape) if frame_read else None
        capture.release()
        results.append(
            {
                "index": index,
                "opened": opened,
                "frame_read": frame_read,
                "frame_shape": frame_shape,
            }
        )
    return results
