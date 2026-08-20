import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "procedural_error_cnn.keras"

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "split_dataset" / "test"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print("   PROCEVISION AI - MODEL EVALUATION")
print("==========================================\n")

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model : {MODEL_PATH}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print(f"Classes : {class_names}")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = (predictions > 0.5).astype(int).flatten()

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(y_true, y_pred)

print("\n==========================================")
print("              RESULTS")
print("==========================================")

print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

print("\n==========================================")
print("         CONFUSION MATRIX")
print("==========================================")

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("       CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# INCORRECT PREDICTIONS
# ============================================================

print("\n==========================================")
print("      INCORRECT PREDICTIONS")
print("==========================================")

incorrect_count = 0

image_index = 0

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = (predictions > 0.5).astype(int).flatten()

    for i in range(len(labels)):

        true_label = int(labels[i].numpy())
        predicted_label = int(predicted_labels[i])

        if true_label != predicted_label:

            print(
                f"Image index {image_index}: "
                f"Actual = {class_names[true_label]}, "
                f"Predicted = {class_names[predicted_label]}"
            )

            incorrect_count += 1

        image_index += 1


print(f"\nIncorrect predictions : {incorrect_count}")

print("\n==========================================")
print("        EVALUATION COMPLETE")
print("==========================================\n")