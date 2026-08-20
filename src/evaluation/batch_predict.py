import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "procedural_error_cnn.keras"
)

ERROR_FRAMES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "error_frames"
    / "16_2"
)

IMAGE_SIZE = (128, 128)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print("   PROCEVISION AI - BATCH PREDICTION")
print("==========================================\n")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# FIND FRAMES
# ============================================================

image_paths = sorted(ERROR_FRAMES_DIR.glob("*.jpg"))

print(f"Error frames found: {len(image_paths)}")


if len(image_paths) == 0:

    print("ERROR: No JPG frames found.")
    raise SystemExit


# ============================================================
# PREDICT
# ============================================================

error_detected = 0
correct_detected = 0


print("\n==========================================")
print("FRAME PREDICTIONS")
print("==========================================")

for image_path in image_paths:

    image = Image.open(image_path).convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0][0]

    if prediction >= 0.5:

        label = "ERROR"
        confidence = prediction
        error_detected += 1

    else:

        label = "CORRECT"
        confidence = 1 - prediction
        correct_detected += 1

    print(
        f"{image_path.name:<45} "
        f"{label:<8} "
        f"{confidence * 100:6.2f}%"
    )


# ============================================================
# SUMMARY
# ============================================================

total = len(image_paths)

print("\n==========================================")
print("BATCH PREDICTION SUMMARY")
print("==========================================")

print(f"Total frames       : {total}")
print(f"Predicted ERROR    : {error_detected}")
print(f"Predicted CORRECT  : {correct_detected}")

if total > 0:

    error_rate = (
        error_detected / total
    ) * 100

    print(
        f"Error detection rate: "
        f"{error_rate:.2f}%"
    )

print("==========================================\n")