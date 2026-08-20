import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path


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


# ============================================================
# SETTINGS
# ============================================================

FRAME_INTERVAL = 2.0       # seconds
SEQUENCE_LENGTH = 8
IMAGE_SIZE = 128
THRESHOLD = 0.5


# ============================================================
# EXTRACT FRAMES
# ============================================================

def extract_frames(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = frame_count / fps

    frames = []
    timestamps = []

    current_time = 0.0

    while current_time < duration:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            current_time * 1000
        )

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = frame.astype("float32") / 255.0

        frames.append(frame)
        timestamps.append(current_time)

        current_time += FRAME_INTERVAL

    cap.release()

    return frames, timestamps


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(frames, timestamps):

    sequences = []
    sequence_times = []

    for start in range(
        0,
        len(frames) - SEQUENCE_LENGTH + 1,
        SEQUENCE_LENGTH
    ):

        sequence = frames[
            start:start + SEQUENCE_LENGTH
        ]

        sequences.append(sequence)

        start_time = timestamps[start]
        end_time = timestamps[
            start + SEQUENCE_LENGTH - 1
        ]

        sequence_times.append(
            (start_time, end_time)
        )

    return (
        np.array(sequences),
        sequence_times
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" VIDEO TEMPORAL PREDICTION")
    print("==========================================")

    # --------------------------------------------------------
    # VIDEO INPUT
    # --------------------------------------------------------

    video_input = input(
        "\nEnter video path: "
    ).strip()

    video_path = Path(video_input)

    if not video_path.exists():

        print("\nERROR: Video not found.")
        return

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading CNN-LSTM model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # EXTRACT FRAMES
    # --------------------------------------------------------

    print("\nExtracting frames...")

    frames, timestamps = extract_frames(
        video_path
    )

    print(
        f"Frames extracted : {len(frames)}"
    )

    # --------------------------------------------------------
    # CREATE SEQUENCES
    # --------------------------------------------------------

    sequences, sequence_times = create_sequences(
        frames,
        timestamps
    )

    print(
        f"Sequences created: {len(sequences)}"
    )

    if len(sequences) == 0:

        print(
            "\nERROR: Video is too short."
        )

        return

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print("\nRunning temporal predictions...")

    probabilities = model.predict(
        sequences,
        verbose=1
    ).flatten()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total = len(predictions)

    error_count = int(
        np.sum(predictions == 1)
    )

    correct_count = total - error_count

    print("\n==========================================")
    print("VIDEO ANALYSIS SUMMARY")
    print("==========================================")

    print(
        f"Video             : {video_path.name}"
    )

    print(
        f"Total sequences   : {total}"
    )

    print(
        f"Predicted CORRECT : {correct_count}"
    )

    print(
        f"Predicted ERROR   : {error_count}"
    )

    # --------------------------------------------------------
    # ERROR REGIONS
    # --------------------------------------------------------

    print("\n==========================================")
    print("DETECTED ERROR REGIONS")
    print("==========================================")

    found_error = False

    for i, prediction in enumerate(predictions):

        if prediction == 1:

            found_error = True

            start_time, end_time = sequence_times[i]

            confidence = probabilities[i]

            print(
                f"{start_time:8.2f}s - "
                f"{end_time:8.2f}s  "
                f"Confidence: {confidence * 100:.2f}%"
            )

    if not found_error:

        print(
            "No procedural error regions detected."
        )

    # --------------------------------------------------------
    # OVERALL RESULT
    # --------------------------------------------------------

    print("\n==========================================")
    print("OVERALL RESULT")
    print("==========================================")

    if error_count > 0:

        print(
            "PROCEDURAL ERROR DETECTED"
        )

    else:

        print(
            "NO PROCEDURAL ERROR DETECTED"
        )

    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()