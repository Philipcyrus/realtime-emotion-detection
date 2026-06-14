"""OpenCV webcam runtime."""

from __future__ import annotations

import time

from emotion_detection.config import RuntimeConfig
from emotion_detection.emotion_model import EmotionClassifier
from emotion_detection.face_detection import HaarFaceDetector
from emotion_detection.models import DetectionResult, FaceBox
from emotion_detection.reporting import export_session_report
from emotion_detection.session import SessionSummary, SessionTracker


WINDOW_NAME = "Emotion Detection"


class EmotionCameraApp:
    """Coordinates webcam capture, face detection, classification, and overlays."""

    def __init__(
        self,
        config: RuntimeConfig,
        detector: HaarFaceDetector,
        classifier: EmotionClassifier,
    ) -> None:
        self._config = config
        self._detector = detector
        self._classifier = classifier

    def run(self) -> SessionSummary:
        import cv2

        capture = self._open_capture()

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.frame_height)

        previous_time = time.perf_counter()
        tracker = SessionTracker(self._config.session_name or "session")

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")

                results = self._detect_emotions(frame)
                current_time = time.perf_counter()
                fps = 1.0 / max(current_time - previous_time, 1e-6)
                previous_time = current_time
                tracker.record_frame(fps)
                for result in results:
                    tracker.record_detection(result)

                self._draw(frame, results, fps)
                if not self._config.headless:
                    cv2.imshow(WINDOW_NAME, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                if (
                    self._config.max_frames is not None
                    and tracker.frames_processed >= self._config.max_frames
                ):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

        summary = tracker.summarize()
        if self._config.report_dir:
            paths = export_session_report(summary, self._config.report_dir)
            print(f"Session report written: {paths['json']} and {paths['html']}")
        return summary

    def _open_capture(self) -> object:
        import cv2

        candidate_indexes = [self._config.camera_index]
        candidate_indexes.extend(
            index for index in range(4) if index != self._config.camera_index
        )

        for index in candidate_indexes:
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                if index != self._config.camera_index:
                    print(
                        f"Webcam index {self._config.camera_index} was unavailable; "
                        f"using index {index} instead."
                    )
                return capture
            capture.release()

        raise RuntimeError(
            "Could not open a webcam at indexes 0-3. Check camera permissions "
            "or pass a working index with --camera-index."
        )

    def _detect_emotions(self, frame: object) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        for face in self._detector.detect(frame):
            crop = frame[face.y:face.bottom, face.x:face.right]
            prediction = self._classifier.predict(crop)
            results.append(DetectionResult(face=face, prediction=prediction))
        return results

    def _draw(self, frame: object, results: list[DetectionResult], fps: float) -> None:
        import cv2

        for result in results:
            face = result.face
            prediction = result.prediction
            label = (
                f"{prediction.label} "
                f"{prediction.confidence:.{self._config.confidence_precision}f} "
                f"({prediction.source})"
            )

            cv2.rectangle(frame, (face.x, face.y), (face.right, face.bottom), (0, 180, 0), 2)
            self._draw_label(frame, face, label)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Press q to quit",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        warning = getattr(self._classifier, "warning", None)
        if warning:
            cv2.putText(
                frame,
                "Starter/demo model: labels are not reliable",
                (16, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_label(self, frame: object, face: FaceBox, label: str) -> None:
        import cv2

        label_y = max(face.y - 10, 24)
        cv2.putText(
            frame,
            label,
            (face.x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
