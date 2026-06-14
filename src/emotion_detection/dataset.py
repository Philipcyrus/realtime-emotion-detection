"""Dataset helpers for training and smoke-test data generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emotion_detection.config import DEFAULT_LABELS


FER2013_HF_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}


def parse_labels(labels: str | None) -> tuple[str, ...]:
    """Parse comma-separated labels or return the default emotion classes."""

    if labels is None:
        return DEFAULT_LABELS

    parsed = tuple(label.strip() for label in labels.split(",") if label.strip())
    if not parsed:
        raise ValueError("At least one label is required.")
    return parsed


def discover_labels(dataset_dir: Path) -> tuple[str, ...]:
    """Return sorted class folder names for an image-folder dataset."""

    labels = tuple(
        path.name for path in sorted(dataset_dir.iterdir()) if path.is_dir()
    )
    if not labels:
        raise ValueError(f"No class folders found in dataset directory: {dataset_dir}")
    return labels


def load_image_folder_arrays(
    dataset_dir: Path,
    labels: tuple[str, ...],
    image_size: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Load an image-folder dataset into grayscale arrays and one-hot labels."""

    import cv2

    images: list[np.ndarray] = []
    targets: list[int] = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for label_index, label in enumerate(labels):
        class_dir = dataset_dir / label
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in extensions:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            images.append(image[..., np.newaxis])
            targets.append(label_index)

    if not images:
        raise ValueError(f"No readable images found in dataset directory: {dataset_dir}")

    x = np.asarray(images, dtype=np.float32)
    y = np.zeros((len(targets), len(labels)), dtype=np.float32)
    y[np.arange(len(targets)), targets] = 1.0
    return x, y


def make_demo_dataset(
    output_dir: Path,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    samples_per_label: int = 8,
    image_size: int = 48,
) -> dict[str, object]:
    """Create a tiny synthetic image-folder dataset for pipeline validation."""

    import cv2

    if samples_per_label < 2:
        raise ValueError("samples_per_label must be at least 2.")

    rng = np.random.default_rng(42)
    splits = ("train", "validation")
    for split in splits:
        for label_index, label in enumerate(labels):
            class_dir = output_dir / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for sample_index in range(samples_per_label):
                image = _synthetic_face(
                    label_index=label_index,
                    sample_index=sample_index,
                    image_size=image_size,
                    rng=rng,
                )
                cv2.imwrite(str(class_dir / f"{label}_{sample_index:03d}.png"), image)

    return {
        "output_dir": str(output_dir),
        "labels": list(labels),
        "samples_per_label": samples_per_label,
        "image_size": image_size,
        "splits": list(splits),
    }


def download_fer2013_from_huggingface(
    output_dir: Path,
    dataset_name: str = "AutumnQiu/fer2013",
    max_per_class: int | None = None,
    splits: tuple[str, ...] = ("train", "valid"),
) -> dict[str, object]:
    """Download FER2013 from Hugging Face into image-folder format."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Run "
            "`./.venv/Scripts/python.exe -m pip install -r requirements.txt`."
        ) from exc

    if max_per_class is not None and max_per_class <= 0:
        raise ValueError("max_per_class must be positive when provided.")

    written: dict[str, dict[str, int]] = {}
    split_map = {"valid": "validation"}

    for split in splits:
        target_split = split_map.get(split, split)
        counts = _existing_image_counts(output_dir / target_split)
        for label in FER2013_HF_LABELS.values():
            counts.setdefault(label, 0)
        if max_per_class is not None and all(
            counts[label] >= max_per_class for label in FER2013_HF_LABELS.values()
        ):
            written[target_split] = counts
            continue

        print(
            f"Downloading split '{split}' into '{target_split}' "
            f"(current counts: {counts})"
        )
        dataset = load_dataset(dataset_name, split=split, streaming=True)

        for row in dataset:
            label = FER2013_HF_LABELS[int(row["label"])]
            if max_per_class is not None and counts[label] >= max_per_class:
                if all(count >= max_per_class for count in counts.values()):
                    break
                continue

            image = row["image"].convert("L")
            class_dir = output_dir / target_split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            image.save(class_dir / f"{target_split}_{label}_{counts[label]:05d}.png")
            counts[label] += 1
            if counts[label] % 100 == 0:
                print(f"{target_split}/{label}: {counts[label]} images")

        written[target_split] = counts

    return {
        "source": dataset_name,
        "output_dir": str(output_dir),
        "labels": list(FER2013_HF_LABELS.values()),
        "max_per_class": max_per_class,
        "splits": written,
    }


def _existing_image_counts(split_dir: Path) -> dict[str, int]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts: dict[str, int] = {}
    for label in FER2013_HF_LABELS.values():
        class_dir = split_dir / label
        if not class_dir.exists():
            counts[label] = 0
            continue
        counts[label] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
    return counts


def _synthetic_face(
    label_index: int,
    sample_index: int,
    image_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    import cv2

    image = np.full((image_size, image_size), 32 + label_index * 18, dtype=np.uint8)
    noise = rng.normal(0, 5, size=image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    center = image_size // 2
    radius = max(12, image_size // 3)
    cv2.circle(image, (center, center), radius, 100 + label_index * 12, thickness=2)

    eye_y = center - image_size // 10
    eye_offset = image_size // 8
    cv2.circle(image, (center - eye_offset, eye_y), 2, 230, thickness=-1)
    cv2.circle(image, (center + eye_offset, eye_y), 2, 230, thickness=-1)

    mouth_y = center + image_size // 7
    mouth_width = image_size // 5
    mood = label_index % 4
    if mood == 0:
        cv2.ellipse(image, (center, mouth_y), (mouth_width, 5), 0, 0, 180, 230, 1)
    elif mood == 1:
        cv2.ellipse(image, (center, mouth_y + 4), (mouth_width, 5), 0, 180, 360, 230, 1)
    elif mood == 2:
        cv2.line(image, (center - mouth_width, mouth_y), (center + mouth_width, mouth_y), 230, 1)
    else:
        cv2.circle(image, (center, mouth_y), 4 + (sample_index % 3), 230, 1)

    return image
