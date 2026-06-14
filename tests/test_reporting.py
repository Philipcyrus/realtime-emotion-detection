import tempfile
import unittest
from pathlib import Path

from emotion_detection.models import DetectionResult, EmotionPrediction, FaceBox
from emotion_detection.reporting import export_session_report
from emotion_detection.session import SessionTracker


class ReportingTests(unittest.TestCase):
    def test_session_report_exports_json_and_html_without_images(self) -> None:
        tracker = SessionTracker("Portfolio Demo")
        tracker.record_frame(30.0)
        tracker.record_detection(
            DetectionResult(
                face=FaceBox(x=1, y=2, width=3, height=4),
                prediction=EmotionPrediction(
                    label="happy",
                    confidence=0.91,
                    source="demo",
                ),
            )
        )
        summary = tracker.summarize()

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = export_session_report(summary, Path(temp_dir))

            self.assertTrue(paths["json"].exists())
            self.assertTrue(paths["html"].exists())
            html = paths["html"].read_text(encoding="utf-8")
            self.assertIn("Emotion Detection Report", html)
            self.assertNotIn("<img", html.lower())


if __name__ == "__main__":
    unittest.main()
