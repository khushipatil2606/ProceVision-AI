import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "procedural_error_cnn_lstm_v2.keras"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step_labeled_split"
    / "test"
)


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    sequences = []
    labels = []
    filenames = []

    for class_name, label in [
        ("correct", 0),
        ("error", 1)
    ]:

        folder = TEST_DIR / class_name

        files = sorted(
            folder.glob("*.npz")
        )

        for file in files:

            data = np.load(
                file,
                allow_pickle=True
            )

            sequences.append(
                data["frames"].astype(
                    np.float32
                )
            )

            labels.append(label)
            filenames.append(
                file.name
            )

    return (
        np.array(sequences),
        np.array(labels),
        filenames
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" CNN-LSTM V2 EVALUATION")
    print("==========================================")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print(
        "\nLoading corrected test dataset..."
    )

    X_test, y_test, filenames = load_data()

    print(
        f"Test sequences : {len(X_test)}"
    )

    print(
        f"Correct        : "
        f"{np.sum(y_test == 0)}"
    )

    print(
        f"Error          : "
        f"{np.sum(y_test == 1)}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print(
        "\nGenerating predictions..."
    )

    scores = model.predict(
        X_test,
        verbose=1
    ).flatten()

    predictions = (
        scores >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = np.mean(
        predictions == y_test
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "correct",
            "error"
        ],
        digits=4
    )

    print("\n==========================================")
    print("RESULTS")
    print("==========================================")

    print(
        f"Test Accuracy : "
        f"{accuracy:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print("\n==========================================")
    print("CONFUSION MATRIX")
    print("==========================================")

    print(cm)

    print("\n==========================================")
    print("CLASSIFICATION REPORT")
    print("==========================================")

    print(report)

    # --------------------------------------------------------
    # INCORRECT PREDICTIONS
    # --------------------------------------------------------

    print("\n==========================================")
    print("INCORRECT PREDICTIONS")
    print("==========================================")

    incorrect = np.where(
        predictions != y_test
    )[0]

    if len(incorrect) == 0:

        print(
            "No incorrect predictions."
        )

    else:

        for index in incorrect:

            actual = (
                "ERROR"
                if y_test[index] == 1
                else "CORRECT"
            )

            predicted = (
                "ERROR"
                if predictions[index] == 1
                else "CORRECT"
            )

            print(
                f"{filenames[index]:<55}"
                f"Actual={actual:<8}"
                f"Predicted={predicted:<8}"
                f"Score={scores[index]:.4f}"
            )

    # --------------------------------------------------------
    # ERROR DETECTION
    # --------------------------------------------------------

    actual_errors = np.sum(
        y_test == 1
    )

    detected_errors = np.sum(
        (y_test == 1)
        &
        (predictions == 1)
    )

    if actual_errors > 0:

        detection_rate = (
            detected_errors
            / actual_errors
            * 100
        )

    else:

        detection_rate = 0

    print("\n==========================================")
    print("ERROR DETECTION")
    print("==========================================")

    print(
        f"Actual error sequences   : "
        f"{actual_errors}"
    )

    print(
        f"Detected error sequences : "
        f"{detected_errors}"
    )

    print(
        f"Error detection rate    : "
        f"{detection_rate:.2f}%"
    )

    print("\n==========================================")
    print("CNN-LSTM V2 EVALUATION COMPLETE")
    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()