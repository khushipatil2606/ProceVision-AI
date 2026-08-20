import tensorflow as tf
import numpy as np

from pathlib import Path
from PIL import Image
import shutil


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "procedural_error_cnn.keras"

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split_dataset"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "misclassified"
)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 1


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print("   PROCEVISION AI - ERROR ANALYSIS")
print("==========================================\n")

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print(f"Classes: {class_names}")


# ============================================================
# GET IMAGE PATHS
# ============================================================

image_paths = []

for class_name in class_names:

    class_dir = TEST_DIR / class_name

    for image_path in sorted(class_dir.glob("*")):

        if image_path.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]:
            image_paths.append(image_path)


# ============================================================
# FIND MISCLASSIFIED IMAGES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

incorrect_count = 0

print("\n==========================================")
print("       MISCLASSIFIED IMAGES")
print("==========================================\n")


for index, (images, labels) in enumerate(test_dataset):

    prediction = model.predict(images, verbose=0)[0][0]

    predicted_label = 1 if prediction >= 0.5 else 0

    actual_label = int(labels.numpy()[0])

    image_path = image_paths[index]

    if actual_label != predicted_label:

        incorrect_count += 1

        confidence = (
            prediction
            if predicted_label == 1
            else 1 - prediction
        )

        print(f"Image       : {image_path.name}")
        print(f"Actual      : {class_names[actual_label]}")
        print(f"Predicted   : {class_names[predicted_label]}")
        print(f"Confidence  : {confidence:.4f}")
        print("-" * 50)

        # Copy image for inspection
        destination = (
            OUTPUT_DIR
            / f"misclassified_{incorrect_count}_{image_path.name}"
        )

        shutil.copy2(image_path, destination)


print("\n==========================================")
print("       ERROR ANALYSIS COMPLETE")
print("==========================================")

print(f"Misclassified images : {incorrect_count}")
print(f"Saved to             : {OUTPUT_DIR}")

print("==========================================\n")