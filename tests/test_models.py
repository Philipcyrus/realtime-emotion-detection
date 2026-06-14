import unittest

from emotion_detection.models import FaceBox


class FaceBoxTests(unittest.TestCase):
    def test_edges_are_derived_from_origin_and_size(self) -> None:
        box = FaceBox(x=10, y=20, width=30, height=40)

        self.assertEqual(box.right, 40)
        self.assertEqual(box.bottom, 60)


if __name__ == "__main__":
    unittest.main()
