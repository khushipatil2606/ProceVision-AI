import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "procedural_error_cnn_lstm.keras"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "annotated_split"
    / "test"
)


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = [
    "correct",
    "error"
]


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_test_sequences():

    X = []
    y = []
    filenames = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_dir = TEST_DIR / class_name

        if not class_dir.exists():
            print(f"WARNING: Missing folder: {class_dir}")
            continue

        files = sorted(
            class_dir.glob("*.npz")
        )

        print(
            f"{class_name.upper()} sequences: "
            f"{len(files)}"
        )

        for file in files:

            data = np.load(file)

            sequence = data["frames"]

            X.append(sequence)
            y.append(label)
            filenames.append(file.name)

    return (
        np.array(X),
        np.array(y),
        filenames
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI - CNN-LSTM EVALUATION")
    print("==========================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")
    print(f"Model : {MODEL_PATH}")

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("\nLoading test sequences...")

    X_test, y_test, filenames = load_test_sequences()

    print("\n==========================================")
    print("TEST DATASET")
    print("==========================================")

    print(
        f"Total sequences : {len(X_test)}"
    )

    print(
        f"Correct         : {np.sum(y_test == 0)}"
    )

    print(
        f"Error           : {np.sum(y_test == 1)}"
    )

    print(
        f"Sequence shape  : {X_test.shape}"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test,
        verbose=1
    )

    # CNN-LSTM output is binary probability
    y_pred = (
        predictions.flatten() >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n==========================================")
    print("RESULTS")
    print("==========================================")

    print(
        f"Test Accuracy : {accuracy:.4f}"
    )

    print(
        f"Test Accuracy : {accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n==========================================")
    print("CONFUSION MATRIX")
    print("==========================================")

    print(cm)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n==========================================")
    print("CLASSIFICATION REPORT")
    print("==========================================")

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=2
    )

    print(report)

    # --------------------------------------------------------
    # INCORRECT PREDICTIONS
    # --------------------------------------------------------

    print("\n==========================================")
    print("INCORRECT PREDICTIONS")
    print("==========================================")

    incorrect = 0

    for i in range(len(y_test)):

        if y_test[i] != y_pred[i]:

            incorrect += 1

            print(
                f"{filenames[i]:45s} "
                f"Actual={CLASS_NAMES[y_test[i]]:<8} "
                f"Predicted={CLASS_NAMES[y_pred[i]]}"
            )

    print(
        f"\nIncorrect predictions : {incorrect}"
    )

    # --------------------------------------------------------
    # ERROR DETECTION
    # --------------------------------------------------------

    actual_errors = np.sum(
        y_test == 1
    )

    detected_errors = np.sum(
        (y_test == 1) &
        (y_pred == 1)
    )

    if actual_errors > 0:

        detection_rate = (
            detected_errors /
            actual_errors
        )

    else:

        detection_rate = 0

    print("\n==========================================")
    print("ERROR DETECTION")
    print("==========================================")

    print(
        f"Actual error sequences   : {actual_errors}"
    )

    print(
        f"Detected error sequences : {detected_errors}"
    )

    print(
        f"Error detection rate    : "
        f"{detection_rate * 100:.2f}%"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n==========================================")
    print("CNN-LSTM EVALUATION COMPLETE")
    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()