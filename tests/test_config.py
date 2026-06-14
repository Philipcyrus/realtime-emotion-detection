import unittest

from emotion_detection.config import DEFAULT_LABELS, RuntimeConfig


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_use_expected_labels(self) -> None:
        config = RuntimeConfig.from_values()

        self.assertEqual(config.camera_index, 0)
        self.assertEqual(config.labels, DEFAULT_LABELS)

    def test_custom_labels_are_trimmed(self) -> None:
        config = RuntimeConfig.from_values(labels="happy, neutral, sad")

        self.assertEqual(config.labels, ("happy", "neutral", "sad"))

    def test_report_config_is_parsed(self) -> None:
        config = RuntimeConfig.from_values(
            report_dir="reports",
            session_name="demo",
            max_frames=10,
        )

        self.assertEqual(str(config.report_dir), "reports")
        self.assertEqual(config.session_name, "demo")
        self.assertEqual(config.max_frames, 10)

    def test_headless_requires_max_frames(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeConfig.from_values(headless=True)

    def test_empty_custom_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeConfig.from_values(labels=", ,")


if __name__ == "__main__":
    unittest.main()
