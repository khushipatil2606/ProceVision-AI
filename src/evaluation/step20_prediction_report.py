import tensorflow as tf
import numpy as np
import pandas as pd

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
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "step20_prediction_report.csv"
)


IMAGE_SIZE = (128, 128)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI - STEP 20 REPORT")
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
    # FIND IMAGES
    # --------------------------------------------------------

    images = sorted(
        ERROR_DIR.glob("*.jpg")
    )

    print(
        f"Error frames found: {len(images)}"
    )


    # --------------------------------------------------------
    # CREATE RESULTS
    # --------------------------------------------------------

    results = []


    for image_path in images:

        image = tf.keras.utils.load_img(
            image_path,
            target_size=IMAGE_SIZE
        )

        image_array = (
            tf.keras.utils.img_to_array(image)
        )

        image_array = image_array / 255.0

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # Prediction
        probability = float(
            model.predict(
                image_array,
                verbose=0
            )[0][0]
        )


        if probability >= 0.5:

            prediction = "ERROR"
            confidence = probability

        else:

            prediction = "CORRECT"
            confidence = 1 - probability


        results.append({
            "image": image_path.name,
            "actual_class": "ERROR",
            "predicted_class": prediction,
            "confidence": round(
                confidence * 100,
                2
            ),
            "raw_score": round(
                probability,
                4
            )
        })


    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    df = pd.DataFrame(results)


    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    error_count = (
        df["predicted_class"]
        .eq("ERROR")
        .sum()
    )

    correct_count = (
        df["predicted_class"]
        .eq("CORRECT")
        .sum()
    )


    detection_rate = (
        error_count / len(df)
    ) * 100


    print("\n==========================================")
    print("STEP 20 PREDICTION REPORT")
    print("==========================================")

    print(
        f"Total images       : {len(df)}"
    )

    print(
        f"Predicted ERROR    : {error_count}"
    )

    print(
        f"Predicted CORRECT  : {correct_count}"
    )

    print(
        f"Detection rate     : "
        f"{detection_rate:.2f}%"
    )

    print("\nReport saved to:")

    print(OUTPUT_FILE)

    print("\n==========================================")
    print("REPORT GENERATION COMPLETE")
    print("==========================================\n")


if __name__ == "__main__":
    main()