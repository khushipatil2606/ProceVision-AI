import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8


# ============================================================
# LOAD MODEL
# ============================================================

print("\n==========================================")
print("   PROCEVISION AI - CONFUSION MATRIX")
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
# GENERATE PREDICTIONS
# ============================================================

y_true = []
y_pred = []

print("\nGenerating predictions...")

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted_labels = (
        predictions.flatten() >= 0.5
    ).astype(int)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot()

plt.title("ProceVision AI - Procedural Error Detection")

plt.tight_layout()

output_path = OUTPUT_DIR / "confusion_matrix.png"

plt.savefig(output_path, dpi=300)

plt.show()

print("\n==========================================")
print("CONFUSION MATRIX SAVED")
print("==========================================")

print(f"Saved to:")
print(output_path)

print("==========================================\n")