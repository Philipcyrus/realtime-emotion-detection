# Portfolio Notes

This project is designed to present as a complete AI engineering portfolio piece.

## Demonstrated Skills

- Real-time computer vision with OpenCV.
- TensorFlow/Keras training and inference.
- Image-folder dataset handling.
- Model metadata and evaluation reports.
- Privacy-safe session analytics.
- JSON and HTML report generation.
- Modular project organization.
- Bash-first developer workflow.

## Demo Story

1. Run `scripts/check.sh` to show the environment is healthy.
2. Run `probe-camera` to show camera discovery.
3. Run the live app with `bash scripts/run_live.sh 0`, replacing `0` with the camera index reported by `probe-camera`.
4. Export a webcam report into `reports/`.
5. Show the training pipeline with `scripts/smoke_pipeline.sh`.
6. Show the real-data path with `download-fer2013`.
7. Explain that the starter model validates wiring, while a real FER-style dataset is required for meaningful accuracy.

## Honest Model Positioning

The generated starter model is intentionally small. It proves the end-to-end pipeline but should not be represented as a production-grade emotion recognizer. For a stronger portfolio result, train the same architecture on a real facial expression dataset and compare accuracy across experiments.

## Real Dataset Path

Use:

```bash
./scripts/build_better_model.cmd 4000 60 64
```

That command downloads more real FER2013-style data, trains the augmented CNN, and writes evaluation results to `reports/evaluation.json`.
