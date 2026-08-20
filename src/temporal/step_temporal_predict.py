import json
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

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_metadata"
    / "recordings.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 128

SEQUENCE_LENGTH = 8

FRAME_INTERVAL = 2.0

THRESHOLD = 0.5


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# PREPROCESS FRAME
# ============================================================

def preprocess_frame(frame):

    frame = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = frame.astype(
        np.float32
    ) / 255.0

    return frame


# ============================================================
# EXTRACT STEP FRAMES
# ============================================================

def extract_step_frames(
    cap,
    start_time,
    end_time
):

    frames = []
    timestamps = []

    current_time = start_time

    while current_time <= end_time:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            current_time * 1000
        )

        success, frame = cap.read()

        if not success:
            break

        frames.append(
            preprocess_frame(frame)
        )

        timestamps.append(
            current_time
        )

        current_time += FRAME_INTERVAL

    return frames, timestamps


# ============================================================
# CREATE STEP SEQUENCES
# ============================================================

def create_sequences(
    frames,
    timestamps
):

    sequences = []
    sequence_times = []

    if len(frames) < SEQUENCE_LENGTH:

        return (
            sequences,
            sequence_times
        )

    # --------------------------------------------------------
    # Sliding window
    # --------------------------------------------------------

    for start in range(
        0,
        len(frames) - SEQUENCE_LENGTH + 1
    ):

        end = (
            start
            + SEQUENCE_LENGTH
        )

        sequence = frames[
            start:end
        ]

        sequences.append(
            np.array(
                sequence,
                dtype=np.float32
            )
        )

        sequence_times.append(
            (
                timestamps[start],
                timestamps[end - 1]
            )
        )

    return (
        sequences,
        sequence_times
    )


# ============================================================
# ANALYZE STEP
# ============================================================

def analyze_step(
    model,
    cap,
    step
):

    start_time = step[
        "start_time"
    ]

    end_time = step[
        "end_time"
    ]

    frames, timestamps = extract_step_frames(
        cap,
        start_time,
        end_time
    )

    if len(frames) < SEQUENCE_LENGTH:

        return {
            "prediction": "INSUFFICIENT_DATA",
            "confidence": 0.0,
            "sequences": 0,
            "error_sequences": 0
        }

    sequences, sequence_times = create_sequences(
        frames,
        timestamps
    )

    if len(sequences) == 0:

        return {
            "prediction": "INSUFFICIENT_DATA",
            "confidence": 0.0,
            "sequences": 0,
            "error_sequences": 0
        }

    sequences = np.array(
        sequences,
        dtype=np.float32
    )

    probabilities = model.predict(
        sequences,
        verbose=0
    ).flatten()

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    error_count = int(
        np.sum(predictions == 1)
    )

    total_count = len(predictions)

    # --------------------------------------------------------
    # Error ratio
    # --------------------------------------------------------

    error_ratio = (
        error_count /
        total_count
    )

    # --------------------------------------------------------
    # Average probability
    # --------------------------------------------------------

    average_probability = float(
        np.mean(probabilities)
    )

    # --------------------------------------------------------
    # Maximum probability
    # --------------------------------------------------------

    maximum_probability = float(
        np.max(probabilities)
    )

    # --------------------------------------------------------
    # Majority vote
    # --------------------------------------------------------

    if error_ratio >= 0.5:

        prediction = "ERROR"

        confidence = error_ratio

    else:

        prediction = "CORRECT"

        confidence = 1.0 - error_ratio

    return {
        "prediction": prediction,
        "confidence": confidence,
        "average_probability": average_probability,
        "maximum_probability": maximum_probability,
        "sequences": total_count,
        "error_sequences": error_count
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" STEP-LEVEL TEMPORAL PREDICTION")
    print("==========================================")

    # ========================================================
    # ASK RECORDING
    # ========================================================

    recording_id = input(
        "\nEnter recording ID (example: 16_2): "
    ).strip()

    # ========================================================
    # LOAD METADATA
    # ========================================================

    print("\nLoading metadata...")

    metadata = load_metadata()

    recording = None

    for item in metadata:

        if item[
            "recording_id"
        ] == recording_id:

            recording = item

            break

    if recording is None:

        print(
            f"\nERROR: Recording {recording_id} "
            f"was not found in metadata."
        )

        return

    # ========================================================
    # INFORMATION
    # ========================================================

    print("\n==========================================")
    print("RECORDING INFORMATION")
    print("==========================================")

    print(
        f"Recording : "
        f"{recording['recording_id']}"
    )

    print(
        f"Known label : "
        f"{recording['label']}"
    )

    print(
        f"Steps : "
        f"{len(recording['steps'])}"
    )

    print(
        f"Video : "
        f"{recording['video']}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading CNN-LSTM model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    video_path = Path(
        recording["video"]
    )

    if not video_path.exists():

        print(
            f"\nERROR: Video not found:\n"
            f"{video_path}"
        )

        return

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            "\nERROR: Could not open video."
        )

        return

    # ========================================================
    # ANALYZE ALL STEPS
    # ========================================================

    results = []

    print("\n==========================================")
    print("STEP ANALYSIS")
    print("==========================================")

    for index, step in enumerate(
        recording["steps"]
    ):

        print(
            f"\nAnalyzing Step "
            f"{index + 1}/"
            f"{len(recording['steps'])}..."
        )

        result = analyze_step(
            model,
            cap,
            step
        )

        result.update({

            "step_number": index + 1,

            "step_id": step.get(
                "step_id"
            ),

            "description": step.get(
                "description",
                ""
            ),

            "start_time": step[
                "start_time"
            ],

            "end_time": step[
                "end_time"
            ]
        })

        results.append(
            result
        )

        print(
            f"Result      : "
            f"{result['prediction']}"
        )

        print(
            f"Confidence  : "
            f"{result['confidence'] * 100:.2f}%"
        )

        print(
            f"Sequences   : "
            f"{result['sequences']}"
        )

    cap.release()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n==========================================")
    print("FINAL STEP REPORT")
    print("==========================================")

    errors_found = []

    for result in results:

        print(
            f"Step {result['step_number']:02d} | "
            f"{result['prediction']:<18} | "
            f"{result['start_time']:.2f}s - "
            f"{result['end_time']:.2f}s"
        )

        if result["prediction"] == "ERROR":

            errors_found.append(
                result
            )

    # ========================================================
    # ERROR SUMMARY
    # ========================================================

    print("\n==========================================")
    print("ERROR SUMMARY")
    print("==========================================")

    if errors_found:

        print(
            f"Potential error steps: "
            f"{len(errors_found)}"
        )

        for result in errors_found:

            print("\n------------------------------------------")

            print(
                f"Step {result['step_number']}"
            )

            print(
                f"Step ID : "
                f"{result['step_id']}"
            )

            print(
                f"Time : "
                f"{result['start_time']:.2f}s - "
                f"{result['end_time']:.2f}s"
            )

            print(
                f"Confidence : "
                f"{result['confidence'] * 100:.2f}%"
            )

            print(
                f"Description : "
                f"{result['description']}"
            )

    else:

        print(
            "No potential procedural errors detected."
        )

    # ========================================================
    # OVERALL RESULT
    # ========================================================

    print("\n==========================================")
    print("OVERALL RESULT")
    print("==========================================")

    if errors_found:

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