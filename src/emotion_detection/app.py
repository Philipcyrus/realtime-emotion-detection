"""Command-line entrypoint for the real-time emotion detection app."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from emotion_detection.camera import EmotionCameraApp
from emotion_detection.config import RuntimeConfig
from emotion_detection.dataset import (
    download_fer2013_from_huggingface,
    make_demo_dataset,
    parse_labels,
)
from emotion_detection.doctor import run_doctor
from emotion_detection.emotion_model import create_classifier
from emotion_detection.evaluation import evaluate_model
from emotion_detection.face_detection import HaarFaceDetector
from emotion_detection.inference import predict_image
from emotion_detection.training import train_model


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time webcam emotion detection.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index.")
    parser.add_argument("--model", help="Path to a TensorFlow/Keras emotion model.")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height.")
    parser.add_argument(
        "--labels",
        help="Comma-separated label order matching the model output.",
    )
    parser.add_argument("--report-dir", help="Write JSON and HTML session reports here.")
    parser.add_argument("--session-name", help="Name used for report files and metadata.")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames for smoke tests.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an OpenCV window; requires --max-frames.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Premium real-time emotion detection project CLI."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check imports, model, and camera availability.")
    doctor_parser.add_argument("--model", default="assets/models/emotion_model.keras")
    doctor_parser.add_argument("--camera-max-index", type=int, default=3)

    run_parser = subparsers.add_parser("run", help="Start webcam emotion detection.")
    _add_run_arguments(run_parser)

    probe_parser = subparsers.add_parser("probe-camera", help="Probe webcam indexes.")
    probe_parser.add_argument("--max-index", type=int, default=3)

    download_parser = subparsers.add_parser(
        "download-fer2013",
        help="Download FER2013 from Hugging Face into image-folder format.",
    )
    download_parser.add_argument("--output", default="data/fer2013_images")
    download_parser.add_argument("--dataset", default="AutumnQiu/fer2013")
    download_parser.add_argument(
        "--max-per-class",
        type=int,
        help="Limit images per class for quick experiments.",
    )
    download_parser.add_argument(
        "--splits",
        default="train,valid",
        help="Comma-separated Hugging Face splits to download.",
    )

    demo_parser = subparsers.add_parser(
        "make-demo-data",
        help="Create a tiny synthetic image dataset for pipeline validation.",
    )
    demo_parser.add_argument("--output", default="data/demo_emotions")
    demo_parser.add_argument("--labels")
    demo_parser.add_argument("--samples-per-label", type=int, default=8)
    demo_parser.add_argument("--image-size", type=int, default=48)

    train_parser = subparsers.add_parser("train", help="Train a Keras emotion model.")
    train_parser.add_argument("--train-dir", required=True)
    train_parser.add_argument("--validation-dir", required=True)
    train_parser.add_argument("--model-out", default="assets/models/emotion_model.keras")
    train_parser.add_argument("--metadata-out")
    train_parser.add_argument("--labels")
    train_parser.add_argument("--epochs", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--image-size", type=int, default=48)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--patience", type=int, default=8)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a saved Keras emotion model.",
    )
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--data-dir", required=True)
    evaluate_parser.add_argument("--output")
    evaluate_parser.add_argument("--metadata")
    evaluate_parser.add_argument("--labels")
    evaluate_parser.add_argument("--batch-size", type=int, default=32)
    evaluate_parser.add_argument("--image-size", type=int, default=48)

    predict_parser = subparsers.add_parser(
        "predict-image",
        help="Predict emotion for a still image using a saved model.",
    )
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--model", default="assets/models/emotion_model.keras")
    predict_parser.add_argument("--labels")
    predict_parser.add_argument(
        "--no-full-image-fallback",
        action="store_true",
        help="Fail if no face is detected instead of classifying the whole image.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command_names = {
        "doctor",
        "run",
        "probe-camera",
        "download-fer2013",
        "make-demo-data",
        "train",
        "evaluate",
        "predict-image",
    }
    if argv and argv[0] in {"-h", "--help"}:
        build_parser().parse_args(argv)
        return
    if not argv or argv[0] not in command_names:
        args = build_run_parser().parse_args(argv)
        _run_webcam(args)
        return

    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = run_doctor(
            model_path=Path(args.model) if args.model else None,
            camera_max_index=args.camera_max_index,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "run":
        _run_webcam(args)
    elif args.command == "probe-camera":
        _probe_camera(args.max_index)
    elif args.command == "download-fer2013":
        splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
        result = download_fer2013_from_huggingface(
            output_dir=Path(args.output),
            dataset_name=args.dataset,
            max_per_class=args.max_per_class,
            splits=splits,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "make-demo-data":
        result = make_demo_dataset(
            output_dir=Path(args.output),
            labels=parse_labels(args.labels),
            samples_per_label=args.samples_per_label,
            image_size=args.image_size,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "predict-image":
        result = predict_image(
            image_path=Path(args.image),
            model_path=Path(args.model),
            labels=parse_labels(args.labels),
            allow_full_image=not args.no_full_image_fallback,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        metadata = train_model(
            train_dir=Path(args.train_dir),
            validation_dir=Path(args.validation_dir),
            model_path=Path(args.model_out),
            metadata_path=Path(args.metadata_out) if args.metadata_out else None,
            labels=args.labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
        )
        print(json.dumps(metadata, indent=2))
    elif args.command == "evaluate":
        result = evaluate_model(
            model_path=Path(args.model),
            data_dir=Path(args.data_dir),
            output_path=Path(args.output) if args.output else None,
            labels=args.labels,
            metadata_path=Path(args.metadata) if args.metadata else None,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        print(json.dumps(result, indent=2))


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index.")
    parser.add_argument("--model", help="Path to a TensorFlow/Keras emotion model.")
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested frame height.")
    parser.add_argument(
        "--labels",
        help="Comma-separated label order matching the model output.",
    )
    parser.add_argument("--report-dir", help="Write JSON and HTML session reports here.")
    parser.add_argument("--session-name", help="Name used for report files and metadata.")
    parser.add_argument("--max-frames", type=int, help="Stop after N frames for smoke tests.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without an OpenCV window; requires --max-frames.",
    )


def _run_webcam(args: argparse.Namespace) -> None:
    config = RuntimeConfig.from_values(
        camera_index=args.camera_index,
        model=args.model,
        width=args.width,
        height=args.height,
        labels=args.labels,
        report_dir=args.report_dir,
        session_name=args.session_name,
        max_frames=args.max_frames,
        headless=args.headless,
    )
    detector = HaarFaceDetector()
    classifier = create_classifier(config.model_path, config.labels)
    EmotionCameraApp(config, detector, classifier).run()


def _probe_camera(max_index: int) -> None:
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
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
