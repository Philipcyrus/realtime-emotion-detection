import tempfile
import unittest
from pathlib import Path

from emotion_detection.dataset import (
    discover_labels,
    download_fer2013_from_huggingface,
    make_demo_dataset,
    parse_labels,
)


class DatasetTests(unittest.TestCase):
    def test_parse_labels_uses_custom_order(self) -> None:
        self.assertEqual(parse_labels("happy, neutral"), ("happy", "neutral"))

    def test_make_demo_dataset_creates_train_and_validation_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "demo"
            result = make_demo_dataset(
                output,
                labels=("happy", "neutral"),
                samples_per_label=2,
            )

            self.assertEqual(result["labels"], ["happy", "neutral"])
            self.assertEqual(discover_labels(output / "train"), ("happy", "neutral"))
            self.assertEqual(len(list((output / "train" / "happy").glob("*.png"))), 2)
            self.assertEqual(len(list((output / "validation" / "neutral").glob("*.png"))), 2)

    def test_download_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            download_fer2013_from_huggingface(Path("unused"), max_per_class=0)


if __name__ == "__main__":
    unittest.main()
