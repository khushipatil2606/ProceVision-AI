import sys
import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "procedural_error_cnn.keras"
)

IMAGE_SIZE = (128, 128)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print("       PROCEVISION AI")
print("     IMAGE ERROR PREDICTOR")
print("==========================================\n")

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# CHECK IMAGE ARGUMENT
# ============================================================

if len(sys.argv) < 2:

    print("\nERROR: Please provide an image path.")

    print("\nExample:")
    print(
        "python src/evaluation/predict_image.py "
        "\"data/processed/dataset/error/example.jpg\""
    )

    sys.exit()


IMAGE_PATH = Path(sys.argv[1])

if not IMAGE_PATH.is_absolute():

    IMAGE_PATH = PROJECT_ROOT / IMAGE_PATH


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():

    print("\nERROR: Image not found:")
    print(IMAGE_PATH)

    sys.exit()


print("\nImage:")
print(IMAGE_PATH)


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

image = image.resize(IMAGE_SIZE)

image_array = np.array(image)

image_array = image_array.astype("float32") / 255.0

image_array = np.expand_dims(image_array, axis=0)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(
    image_array,
    verbose=0
)[0][0]


# ============================================================
# INTERPRET RESULT
# ============================================================

if prediction >= 0.5:

    label = "ERROR"
    confidence = prediction

else:

    label = "CORRECT"
    confidence = 1 - prediction


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n==========================================")
print("             PREDICTION")
print("==========================================")

print(f"Result      : {label}")
print(f"Confidence  : {confidence * 100:.2f}%")
print(f"Raw Score   : {prediction:.4f}")

print("==========================================\n")