import tensorflow as tf
import numpy as np

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "step20_cnn.keras"
)


# ============================================================
# TEST DATA
# ============================================================

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20_split"
    / "test"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 8


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - STEP 20 EVALUATION")
    print("==========================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading model...")

    if not MODEL_PATH.exists():

        print("ERROR: Model not found:")
        print(MODEL_PATH)

        return

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")
    print(f"Model : {MODEL_PATH}")


    # --------------------------------------------------------
    # LOAD TEST DATA
    # --------------------------------------------------------

    print("\nLoading Step 20 test dataset...")

    test_ds = tf.keras.utils.image_dataset_from_directory(

        TEST_DIR,

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        class_names=[
            "correct",
            "error"
        ],

        shuffle=False
    )

    print(
        f"Classes : {test_ds.class_names}"
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = model.predict(
        test_ds,
        verbose=1
    )

    # Convert sigmoid probabilities to classes
    predicted_classes = (
        predictions.flatten() >= 0.5
    ).astype(int)


    # --------------------------------------------------------
    # TRUE LABELS
    # --------------------------------------------------------

    true_classes = np.concatenate(
        [
            labels.numpy()
            for _, labels in test_ds
        ]
    )


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = np.mean(
        predicted_classes == true_classes
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n==========================================")
    print("              RESULTS")
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
        true_classes,
        predicted_classes
    )

    print("\n==========================================")
    print("         CONFUSION MATRIX")
    print("==========================================")

    print(cm)


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\n==========================================")
    print("       CLASSIFICATION REPORT")
    print("==========================================")

    print(
        classification_report(
            true_classes,
            predicted_classes,
            target_names=[
                "correct",
                "error"
            ],
            digits=2
        )
    )


    # --------------------------------------------------------
    # INCORRECT PREDICTIONS
    # --------------------------------------------------------

    print("\n==========================================")
    print("      INCORRECT PREDICTIONS")
    print("==========================================")

    incorrect_count = 0

    image_index = 0

    for images, labels in test_ds:

        batch_predictions = model.predict(
            images,
            verbose=0
        ).flatten()

        batch_predictions = (
            batch_predictions >= 0.5
        ).astype(int)

        for i in range(len(labels)):

            actual = int(
                labels[i].numpy()
            )

            predicted = int(
                batch_predictions[i]
            )

            if actual != predicted:

                print(
                    f"Image index {image_index}: "
                    f"Actual = "
                    f"{test_ds.class_names[actual]}, "
                    f"Predicted = "
                    f"{test_ds.class_names[predicted]}"
                )

                incorrect_count += 1

            image_index += 1


    print(
        f"\nIncorrect predictions : "
        f"{incorrect_count}"
    )

    print("\n==========================================")
    print("STEP 20 EVALUATION COMPLETE")
    print("==========================================\n")


if __name__ == "__main__":
    main()