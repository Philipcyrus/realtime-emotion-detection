# Emotion Detection App

Real-time emotion detection built with Python, OpenCV, TensorFlow, and Keras.

The project now covers the full portfolio loop:

- Probe available webcams.
- Check local environment health with `doctor`.
- Generate a tiny demo dataset for pipeline validation.
- Train a Keras emotion model from image folders.
- Evaluate a saved model.
- Predict emotion from a still image.
- Run real-time webcam inference.
- Export privacy-safe JSON and HTML session reports.

The app does **not** store webcam frames by default. Reports contain metrics only.

Dataset source option: this project can import the FER2013-style Hugging Face dataset `AutumnQiu/fer2013` into the folder format used by the trainer.

## Project Structure

```text
src/emotion_detection/
tests/
scripts/
docs/
assets/models/
reports/
data/
```

## Setup

Use these commands from Git Bash:

```bash
cd /e/work/Emotion_detection
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install --no-build-isolation -e .
```

## Activate The Virtual Environment

From Git Bash, source the helper script:

```bash
source scripts/activate_env.sh
```

After that, you can use shorter commands:

```bash
python -m pip check
python -m unittest discover -s tests
python -m emotion_detection.app --help
```

To deactivate:

```bash
deactivate_env
```

## Verify The Environment

```bash
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -m unittest discover -s tests
./.venv/Scripts/python.exe -m emotion_detection.app --help
./.venv/Scripts/python.exe -m emotion_detection.app doctor --model assets/models/emotion_model.keras
```

Or run the bundled check script:

```bash
bash scripts/check.sh
```

## Simple Commands

If your Git Bash says `bash: command not found`, use the `.cmd` wrappers. They work from Git Bash and do not need Unix tools:

```bash
./scripts/download_data.cmd
./scripts/train_model.cmd
./scripts/evaluate_model.cmd
./scripts/run_app.cmd
```

To rebuild a better model in one command:

```bash
./scripts/build_better_model.cmd
```

Optional quick versions:

```bash
./scripts/download_data.cmd 100
./scripts/train_model.cmd 5
./scripts/run_app.cmd 0
```

If your Git Bash is healthy, these `.sh` wrappers also work:

```bash
bash scripts/download_data.sh
bash scripts/train_model.sh
bash scripts/evaluate_model.sh
bash scripts/run_app.sh
```

Optional quick versions:

```bash
bash scripts/download_data.sh 100
bash scripts/train_model.sh 5
bash scripts/run_app.sh 0
```

Arguments:

- `build_better_model.cmd 4000 60 64` downloads more data, trains, and evaluates.
- `download_data.cmd 4000` downloads up to 4000 images per emotion class.
- `train_model.cmd 60 64` trains for up to 60 epochs with batch size 64 and early stopping.
- `run_app.cmd 0` opens camera index 0.

If your current model is around 48% validation accuracy, rebuild it with:

```bash
./scripts/build_better_model.cmd 4000 60 64
```

## Probe Your Camera

```bash
./.venv/Scripts/python.exe -m emotion_detection.app probe-camera --max-index 3
```

Use whichever index shows `"opened": true`. In your latest probe output, that was camera index `0`.

## Run Webcam Detection

```bash
./.venv/Scripts/python.exe -m emotion_detection.app run \
  --camera-index 0 \
  --model assets/models/emotion_model.keras \
  --report-dir reports \
  --session-name live-demo
```

Press `q` or `Esc` to exit.

The repository includes a generated starter model at `assets/models/emotion_model.keras` after the setup workflow has been run. It proves the application path works, but it is not a high-accuracy model.

## Why The Expression May Be Wrong

If the app shows your face but the emotion label is incorrect, the camera path is working. The problem is model quality.

The included `assets/models/emotion_model.keras` is a generated starter model trained on synthetic images, not a real facial-expression dataset. It exists so the complete application can run immediately. For meaningful predictions, train the model with real labeled face images.

Use the live app to verify:

```bash
bash scripts/run_live.sh 0
```

Then replace the starter model by training on a real FER-style dataset:

```bash
./scripts/build_better_model.cmd 4000 60 64
```

Or use the live helper script:

```bash
bash scripts/run_live.sh 0
```

## End-To-End Smoke Workflow

This proves the complete data -> train -> evaluate -> webcam report pipeline with a tiny generated dataset.

```bash
./.venv/Scripts/python.exe -m emotion_detection.app make-demo-data \
  --output data/smoke_emotions \
  --labels happy,neutral \
  --samples-per-label 3
```

```bash
./.venv/Scripts/python.exe -m emotion_detection.app train \
  --train-dir data/smoke_emotions/train \
  --validation-dir data/smoke_emotions/validation \
  --model-out assets/models/smoke_emotion_model.keras \
  --metadata-out assets/models/smoke_emotion_model.metadata.json \
  --labels happy,neutral \
  --epochs 1 \
  --batch-size 2
```

```bash
./.venv/Scripts/python.exe -m emotion_detection.app evaluate \
  --model assets/models/smoke_emotion_model.keras \
  --data-dir data/smoke_emotions/validation \
  --metadata assets/models/smoke_emotion_model.metadata.json \
  --output reports/smoke_evaluation.json \
  --batch-size 2
```

```bash
./.venv/Scripts/python.exe -m emotion_detection.app \
  --camera-index 0 \
  --model assets/models/smoke_emotion_model.keras \
  --labels happy,neutral \
  --headless \
  --max-frames 3 \
  --report-dir reports \
  --session-name smoke-webcam
```

The smoke model is only for pipeline validation. For real emotion accuracy, train with a real facial-expression dataset such as FER-style image folders.

You can run the same smoke workflow with:

```bash
bash scripts/smoke_pipeline.sh 0
```

This smoke workflow intentionally uses `--headless`, so it will not open a camera window. It only verifies the pipeline and writes `reports/smoke-webcam.json` plus `reports/smoke-webcam.html`.

## Predict A Still Image

```bash
./.venv/Scripts/python.exe -m emotion_detection.app predict-image \
  --image data/starter_emotions/validation/happy/happy_000.png \
  --model assets/models/emotion_model.keras
```

## Train With A Real Dataset

You can download FER2013-style facial expression data directly from Hugging Face:

```bash
./scripts/download_data.cmd 4000
```

For a tiny command check only:

```bash
./.venv/Scripts/python.exe -m emotion_detection.app download-fer2013 \
  --output data/fer2013_sample \
  --max-per-class 2
```

The importer writes this structure:

Expected folder format:

```text
data/fer2013_images/
  train/
    angry/
    disgust/
    fear/
    happy/
    sad/
    surprise/
    neutral/
  validation/
    angry/
    disgust/
    fear/
    happy/
    sad/
    surprise/
    neutral/
```

Train:

```bash
./scripts/train_model.cmd 60 64
```

Evaluate:

```bash
./scripts/evaluate_model.cmd
```

Run with the trained model:

```bash
./scripts/run_app.cmd 0
```

If Hugging Face rate-limits or times out, rerun the `download-fer2013` command. It writes image files locally, so already downloaded files remain on disk.

## Commands

```bash
./.venv/Scripts/python.exe -m emotion_detection.app probe-camera --max-index 3
./.venv/Scripts/python.exe -m emotion_detection.app doctor --model assets/models/emotion_model.keras
./.venv/Scripts/python.exe -m emotion_detection.app download-fer2013 --output data/fer2013_sample --max-per-class 2
./.venv/Scripts/python.exe -m emotion_detection.app make-demo-data --output data/smoke_emotions
./.venv/Scripts/python.exe -m emotion_detection.app train --help
./.venv/Scripts/python.exe -m emotion_detection.app evaluate --help
./.venv/Scripts/python.exe -m emotion_detection.app predict-image --help
./.venv/Scripts/python.exe -m emotion_detection.app run --camera-index 0 --model assets/models/emotion_model.keras
```

## Project Notes

The larger InterviewIQ idea can use this project as its webcam emotion-analysis subsystem.

See `docs/PORTFOLIO.md` for a concise demo narrative and honest model-positioning notes.
See `CHANGELOG.md` for release notes and future project updates.
