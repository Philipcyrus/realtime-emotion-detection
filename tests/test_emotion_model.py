import json
import tempfile
import unittest
from pathlib import Path

from emotion_detection.emotion_model import labels_from_metadata


class EmotionModelMetadataTests(unittest.TestCase):
    def test_labels_are_loaded_from_sibling_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "emotion_model.keras"
            metadata_path = Path(temp_dir) / "emotion_model.metadata.json"
            metadata_path.write_text(
                json.dumps({"labels": ["happy", "neutral"]}),
                encoding="utf-8",
            )

            self.assertEqual(labels_from_metadata(model_path), ("happy", "neutral"))


if __name__ == "__main__":
    unittest.main()
