import tensorflow as tf
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "step20_cnn.keras"
)


TEST_ERROR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20_split"
    / "test"
    / "error"
)


REAL_ERROR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "error_frames"
    / "16_2"
)


IMAGE_SIZE = (128, 128)


# ============================================================
# PREDICT FUNCTION
# ============================================================

def predict_image(model, image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    score = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    return score


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI - STEP 20 DIAGNOSTIC")
    print("==========================================")

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")


    # ========================================================
    # TEST ERROR IMAGES
    # ========================================================

    test_images = sorted(
        TEST_ERROR_DIR.glob("*.jpg")
    )

    print("\n==========================================")
    print("STEP 20 TEST ERROR IMAGES")
    print("==========================================")

    print(
        f"Images found: {len(test_images)}"
    )

    for image_path in test_images:

        score = predict_image(
            model,
            image_path
        )

        if score >= 0.5:
            prediction = "ERROR"
        else:
            prediction = "CORRECT"

        print(
            f"{image_path.name:45} "
            f"{prediction:8} "
            f"score={score:.4f}"
        )


    # ========================================================
    # REAL ERROR FRAMES
    # ========================================================

    real_images = sorted(
        REAL_ERROR_DIR.glob("*.jpg")
    )

    print("\n==========================================")
    print("ORIGINAL ERROR FRAMES")
    print("==========================================")

    print(
        f"Images found: {len(real_images)}"
    )

    error_predictions = 0
    correct_predictions = 0

    for image_path in real_images:

        score = predict_image(
            model,
            image_path
        )

        if score >= 0.5:

            prediction = "ERROR"
            error_predictions += 1

        else:

            prediction = "CORRECT"
            correct_predictions += 1


        print(
            f"{image_path.name:45} "
            f"{prediction:8} "
            f"score={score:.4f}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==========================================")
    print("DIAGNOSTIC SUMMARY")
    print("==========================================")

    print(
        f"Original error frames : {len(real_images)}"
    )

    print(
        f"Predicted ERROR       : {error_predictions}"
    )

    print(
        f"Predicted CORRECT     : {correct_predictions}"
    )

    if len(real_images) > 0:

        rate = (
            error_predictions
            / len(real_images)
        ) * 100

        print(
            f"Error detection rate  : {rate:.2f}%"
        )

    print("\n==========================================")
    print("DIAGNOSTIC COMPLETE")
    print("==========================================\n")


if __name__ == "__main__":
    main()