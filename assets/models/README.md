# Model Assets

Place trained TensorFlow/Keras emotion models in this folder.

Recommended MVP file:

```text
assets/models/emotion_model.keras
```

A starter model can be generated at this path with the project training command. It validates wiring only; train on a real emotion dataset for meaningful accuracy.

The app expects:

- Input: grayscale face crop, 48x48 pixels
- Shape: `(1, 48, 48, 1)`
- Output: class probabilities
- Default label order: angry, disgust, fear, happy, sad, surprise, neutral

Large model files are ignored by git through `.gitignore`.

## Smoke Model

You can generate a tiny pipeline-validation model with:

```bash
./.venv/Scripts/python.exe -m emotion_detection.app make-demo-data --output data/smoke_emotions --labels happy,neutral --samples-per-label 3
./.venv/Scripts/python.exe -m emotion_detection.app train --train-dir data/smoke_emotions/train --validation-dir data/smoke_emotions/validation --model-out assets/models/smoke_emotion_model.keras --metadata-out assets/models/smoke_emotion_model.metadata.json --labels happy,neutral --epochs 1 --batch-size 2
```

That model proves the project pipeline works. It is not a real emotion model.
