import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = [
    "correct",
    "error"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data():

    X = []
    y = []

    for label, class_name in enumerate(CLASS_NAMES):

        class_dir = TEST_DIR / class_name

        files = sorted(
            class_dir.glob("*.npz")
        )

        for file in files:

            data = np.load(file)

            X.append(
                data["frames"]
            )

            y.append(label)

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI - CNN-LSTM REPORT")
    print("==========================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("\nLoading test dataset...")

    X_test, y_test = load_test_data()

    print(
        f"Test sequences : {len(X_test)}"
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    probabilities = model.predict(
        X_test,
        verbose=1
    )

    y_pred = (
        probabilities.flatten() >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n==========================================")
    print("FINAL MODEL PERFORMANCE")
    print("==========================================")

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print("\nClassification Report:")
    print(report)

    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nGenerating confusion matrix...")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot()

    plt.title(
        "ProceVision AI - CNN-LSTM Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "Actual Label"
    )

    confusion_path = (
        OUTPUT_DIR
        / "cnn_lstm_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE TEXT REPORT
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "cnn_lstm_performance_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PROCEVISION AI - CNN-LSTM PERFORMANCE REPORT\n"
        )

        file.write(
            "============================================\n\n"
        )

        file.write(
            f"Test sequences : {len(X_test)}\n"
        )

        file.write(
            f"Correct        : {np.sum(y_test == 0)}\n"
        )

        file.write(
            f"Error          : {np.sum(y_test == 1)}\n\n"
        )

        file.write(
            f"Accuracy       : {accuracy * 100:.2f}%\n\n"
        )

        file.write(
            "CONFUSION MATRIX\n"
        )

        file.write(
            "----------------\n"
        )

        file.write(
            str(cm)
        )

        file.write(
            "\n\nCLASSIFICATION REPORT\n"
        )

        file.write(
            "---------------------\n"
        )

        file.write(
            report
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n==========================================")
    print("REPORT GENERATED")
    print("==========================================")

    print(
        f"Confusion matrix:\n{confusion_path}"
    )

    print(
        f"\nPerformance report:\n{report_path}"
    )

    print("\n==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()