import numpy as np
from pathlib import Path

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
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
    / "procedural_error_cnn_lstm_v3.keras"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "balanced_split"
    / "test"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 8
IMAGE_SIZE = 128
CHANNELS = 3

PREDICTION_THRESHOLD = 0.50


# ============================================================
# LOAD SEQUENCES
# ============================================================

def load_sequences():

    X = []
    y = []
    names = []

    print("\nLoading test sequences...")

    for label_name, label in [
        ("correct", 0),
        ("error", 1)
    ]:

        folder = TEST_DIR / label_name

        if not folder.exists():

            print(
                f"WARNING: Folder not found: {folder}"
            )

            continue

        files = sorted(
            folder.glob("*.npz")
        )

        print(
            f"Loading {label_name}: "
            f"{len(files)} sequences"
        )

        for file in files:

            try:

                with np.load(
                    file,
                    allow_pickle=False
                ) as data:

                    # ------------------------------------------------
                    # CURRENT DATASET FORMAT
                    # ------------------------------------------------

                    if "frames" in data.files:

                        sequence = data["frames"]

                    # ------------------------------------------------
                    # BACKWARD COMPATIBILITY
                    # ------------------------------------------------

                    elif "sequence" in data.files:

                        sequence = data["sequence"]

                    else:

                        print(
                            f"Skipping {file.name}: "
                            f"frames/sequence key not found. "
                            f"Available keys: {data.files}"
                        )

                        continue

                # ----------------------------------------------------
                # CHECK SHAPE
                # ----------------------------------------------------

                expected_shape = (
                    SEQUENCE_LENGTH,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    CHANNELS
                )

                if sequence.shape != expected_shape:

                    print(
                        f"Skipping invalid shape: "
                        f"{file.name} "
                        f"shape={sequence.shape} "
                        f"expected={expected_shape}"
                    )

                    continue

                # ----------------------------------------------------
                # CHECK FOR INVALID VALUES
                # ----------------------------------------------------

                if not np.isfinite(sequence).all():

                    print(
                        f"Skipping invalid values: "
                        f"{file.name}"
                    )

                    continue

                # ----------------------------------------------------
                # STORE
                # ----------------------------------------------------

                X.append(
                    sequence.astype(
                        np.float32
                    )
                )

                y.append(label)

                names.append(
                    file.name
                )

            except Exception as e:

                print(
                    f"Error loading "
                    f"{file.name}: {e}"
                )

    if not X:

        return (
            np.empty(
                (
                    0,
                    SEQUENCE_LENGTH,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    CHANNELS
                ),
                dtype=np.float32
            ),
            np.empty(
                (0,),
                dtype=np.int32
            ),
            []
        )

    return (
        np.stack(X).astype(
            np.float32
        ),
        np.array(
            y,
            dtype=np.int32
        ),
        names
    )


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_sequences(X):

    if len(X) == 0:

        return X

    X = X.astype(
        np.float32
    )

    maximum = np.max(X)

    minimum = np.min(X)

    print("\nData range before normalization:")
    print(
        f"Minimum : {minimum:.4f}"
    )
    print(
        f"Maximum : {maximum:.4f}"
    )

    # ------------------------------------------------------------
    # Convert 0-255 images to 0-1
    # ------------------------------------------------------------

    if maximum > 1.0:

        X = X / 255.0

        print(
            "Normalization : 0-255 -> 0-1"
        )

    else:

        print(
            "Normalization : Already 0-1"
        )

    return X


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" CNN-LSTM V3 EVALUATION")
    print("==========================================")

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not MODEL_PATH.exists():

        print(
            "\nERROR: Model file not found:"
        )

        print(
            MODEL_PATH
        )

        return

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading model...")

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    print(
        "Model loaded successfully."
    )

    print(
        f"Model : {MODEL_PATH}"
    )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    X_test, y_test, names = (
        load_sequences()
    )

    print("\n==========================================")
    print("TEST DATASET")
    print("==========================================")

    print(
        f"Total sequences : "
        f"{len(X_test)}"
    )

    print(
        f"Correct         : "
        f"{np.sum(y_test == 0)}"
    )

    print(
        f"Error           : "
        f"{np.sum(y_test == 1)}"
    )

    # --------------------------------------------------------
    # STOP IF EMPTY
    # --------------------------------------------------------

    if len(X_test) == 0:

        print(
            "\nERROR: No valid test sequences found."
        )

        return

    # ========================================================
    # DATA SHAPE
    # ========================================================

    print(
        f"Sequence shape  : "
        f"{X_test.shape}"
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    X_test = normalize_sequences(
        X_test
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    print("\nGenerating predictions...")

    scores = model.predict(
        X_test,
        verbose=1
    ).reshape(-1)

    predictions = (
        scores >= PREDICTION_THRESHOLD
    ).astype(int)

    # ========================================================
    # BASIC RESULTS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # ========================================================
    # ERROR CLASS METRICS
    # ========================================================

    error_precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    error_recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    error_f1 = f1_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n==========================================")
    print("RESULTS")
    print("==========================================")

    print(
        f"Test Accuracy       : "
        f"{accuracy:.4f}"
    )

    print(
        f"Test Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy   : "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Error Precision     : "
        f"{error_precision * 100:.2f}%"
    )

    print(
        f"Error Recall        : "
        f"{error_recall * 100:.2f}%"
    )

    print(
        f"Error F1-Score      : "
        f"{error_f1 * 100:.2f}%"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n==========================================")
    print("CONFUSION MATRIX")
    print("==========================================")

    print(
        "                 Predicted"
    )

    print(
        "              Correct  Error"
    )

    print(
        f"Actual Correct   "
        f"{tn:6d}  {fp:5d}"
    )

    print(
        f"Actual Error     "
        f"{fn:6d}  {tp:5d}"
    )

    print("\nRaw matrix:")

    print(cm)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n==========================================")
    print("CLASSIFICATION REPORT")
    print("==========================================")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "correct",
                "error"
            ],
            zero_division=0
        )
    )

    # ========================================================
    # INCORRECT PREDICTIONS
    # ========================================================

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
                f"{names[index]:55s} "
                f"Actual={actual:7s} "
                f"Predicted={predicted:7s} "
                f"Score={scores[index]:.4f}"
            )

    print(
        f"\nIncorrect predictions : "
        f"{len(incorrect)}"
    )

    # ========================================================
    # ERROR DETECTION
    # ========================================================

    actual_errors = np.sum(
        y_test == 1
    )

    detected_errors = np.sum(
        (y_test == 1)
        &
        (predictions == 1)
    )

    missed_errors = np.sum(
        (y_test == 1)
        &
        (predictions == 0)
    )

    false_error_alerts = np.sum(
        (y_test == 0)
        &
        (predictions == 1)
    )

    if actual_errors > 0:

        detection_rate = (
            detected_errors
            / actual_errors
        )

    else:

        detection_rate = 0.0

    # ========================================================
    # ERROR DETECTION REPORT
    # ========================================================

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
        f"Missed error sequences   : "
        f"{missed_errors}"
    )

    print(
        f"False error alerts       : "
        f"{false_error_alerts}"
    )

    print(
        f"Error detection rate    : "
        f"{detection_rate * 100:.2f}%"
    )

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    predicted_correct = np.sum(
        predictions == 0
    )

    predicted_error = np.sum(
        predictions == 1
    )

    print("\n==========================================")
    print("PREDICTION DISTRIBUTION")
    print("==========================================")

    print(
        f"Predicted CORRECT : "
        f"{predicted_correct}"
    )

    print(
        f"Predicted ERROR   : "
        f"{predicted_error}"
    )

    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print("\n==========================================")
    print("MODEL INTERPRETATION")
    print("==========================================")

    if error_recall >= 0.90:

        print(
            "GOOD: The model detects most "
            "procedural errors."
        )

    elif error_recall >= 0.70:

        print(
            "MODERATE: The model detects "
            "many procedural errors, "
            "but some are missed."
        )

    else:

        print(
            "WARNING: Error detection is weak. "
            "The model is missing many "
            "procedural errors."
        )

    if predicted_error == 0:

        print(
            "WARNING: The model predicted "
            "NO ERROR sequences."
        )

        print(
            "Accuracy alone should NOT be "
            "considered reliable."
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n==========================================")
    print("CNN-LSTM V3 EVALUATION COMPLETE")
    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()