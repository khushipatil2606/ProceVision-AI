import tensorflow as tf
import numpy as np

from pathlib import Path


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
# ERROR FRAMES
# ============================================================

ERROR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "error_frames"
    / "16_2"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI - STEP 20 ERROR TEST")
    print("==========================================")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading Step 20 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")


    # --------------------------------------------------------
    # FIND ERROR FRAMES
    # --------------------------------------------------------

    images = sorted(
        ERROR_DIR.glob("*.jpg")
    )

    print(
        f"\nError frames found: {len(images)}"
    )


    if len(images) == 0:

        print("ERROR: No images found.")

        return


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    predicted_error = 0
    predicted_correct = 0


    print("\n==========================================")
    print("FRAME PREDICTIONS")
    print("==========================================")


    for image_path in images:

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        image_array = (
            tf.keras.utils.img_to_array(
                image
            )
        )

        image_array = (
            image_array / 255.0
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        probability = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )


        if probability >= 0.5:

            prediction = "ERROR"

            confidence = probability

            predicted_error += 1

        else:

            prediction = "CORRECT"

            confidence = 1 - probability

            predicted_correct += 1


        print(
            f"{image_path.name:<45}"
            f"{prediction:<10}"
            f"{confidence * 100:.2f}%"
        )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total = len(images)

    detection_rate = (
        predicted_error / total
    )


    print("\n==========================================")
    print("STEP 20 ERROR TEST SUMMARY")
    print("==========================================")

    print(
        f"Total frames        : {total}"
    )

    print(
        f"Predicted ERROR     : "
        f"{predicted_error}"
    )

    print(
        f"Predicted CORRECT   : "
        f"{predicted_correct}"
    )

    print(
        f"Error detection rate: "
        f"{detection_rate * 100:.2f}%"
    )

    print("==========================================\n")


if __name__ == "__main__":
    main()