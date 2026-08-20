import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model


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


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 8
FRAME_INTERVAL = 2.0
IMAGE_SIZE = 128

# Model output is treated as:
#     probability of ERROR
ERROR_THRESHOLD = 0.50

# Isolated predictions below this confidence are ignored.
ISOLATED_ERROR_MIN_CONFIDENCE = 0.70


# ============================================================
# LOAD MODEL
# ============================================================

def load_cnn_lstm_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    return load_model(
        MODEL_PATH,
        compile=False,
    )


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        cap.release()
        raise RuntimeError(
            "Could not determine video FPS."
        )

    duration = total_video_frames / fps

    frames = []
    timestamps = []

    current_time = 0.0

    while current_time < duration:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            current_time * 1000,
        )

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(
            frame,
            (IMAGE_SIZE, IMAGE_SIZE),
        )

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)
        timestamps.append(current_time)

        current_time += FRAME_INTERVAL

    cap.release()

    return (
        np.asarray(frames, dtype=np.float32),
        np.asarray(timestamps, dtype=np.float32),
        total_video_frames,
        duration,
    )


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================

def create_sequences(frames, timestamps):
    sequences = []
    sequence_times = []

    total_frames = len(frames)

    # Non-overlapping groups of 8 sampled frames.
    for start in range(
        0,
        total_frames - SEQUENCE_LENGTH + 1,
        SEQUENCE_LENGTH,
    ):
        end = start + SEQUENCE_LENGTH

        sequence = frames[start:end]

        start_time = float(timestamps[start])
        end_time = float(timestamps[end - 1])

        sequences.append(sequence)

        sequence_times.append(
            (start_time, end_time)
        )

    if not sequences:
        return (
            np.empty(
                (
                    0,
                    SEQUENCE_LENGTH,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    3,
                ),
                dtype=np.float32,
            ),
            [],
        )

    return (
        np.asarray(sequences, dtype=np.float32),
        sequence_times,
    )


# ============================================================
# TEMPORAL CONSISTENCY
# ============================================================

def apply_temporal_consistency(
    raw_predictions,
    scores,
):
    """
    Keeps strong isolated errors, but removes weak
    one-off predictions.

    This does NOT change strong consecutive errors.
    """

    predictions = raw_predictions.copy()

    for i in range(len(raw_predictions)):
        if raw_predictions[i] != 1:
            continue

        previous_error = (
            i > 0
            and raw_predictions[i - 1] == 1
        )

        next_error = (
            i < len(raw_predictions) - 1
            and raw_predictions[i + 1] == 1
        )

        if (
            not previous_error
            and not next_error
            and float(scores[i])
            < ISOLATED_ERROR_MIN_CONFIDENCE
        ):
            predictions[i] = 0

    return predictions


# ============================================================
# MERGE CONTINUOUS ERROR EVENTS
# ============================================================

def merge_error_regions(regions):
    """
    Example:

        992-1006
        1008-1022
        1024-1038
        1040-1054
        1056-1070

    becomes:

        992-1070

    This is why the UI reports ONE continuous error event
    instead of five separate error events.
    """

    if not regions:
        return []

    regions = sorted(
        regions,
        key=lambda x: x["start"],
    )

    merged = [
        regions[0].copy()
    ]

    max_allowed_gap = FRAME_INTERVAL + 0.01

    for current in regions[1:]:
        previous = merged[-1]

        gap = (
            current["start"]
            - previous["end"]
        )

        if gap <= max_allowed_gap:
            previous["end"] = max(
                previous["end"],
                current["end"],
            )

            previous["confidence"] = max(
                previous["confidence"],
                current["confidence"],
            )

            previous["sequence_end"] = (
                current.get(
                    "sequence_end",
                    previous.get("sequence_end"),
                )
            )

        else:
            merged.append(
                current.copy()
            )

    for event in merged:
        event["duration"] = max(
            0.0,
            event["end"] - event["start"],
        )

    return merged


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze_video(video_path):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_cnn_lstm_model()

    # --------------------------------------------------------
    # EXTRACT FRAMES
    # --------------------------------------------------------

    (
        frames,
        timestamps,
        original_frame_count,
        duration,
    ) = extract_frames(video_path)

    if len(frames) < SEQUENCE_LENGTH:
        raise RuntimeError(
            "Not enough sampled frames to create "
            "a temporal sequence."
        )

    # --------------------------------------------------------
    # CREATE SEQUENCES
    # --------------------------------------------------------

    sequences, sequence_times = create_sequences(
        frames,
        timestamps,
    )

    if len(sequences) == 0:
        raise RuntimeError(
            "No temporal sequences were created."
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    scores = model.predict(
        sequences,
        verbose=0,
    ).reshape(-1)

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    # Safety clamp.
    scores = np.clip(
        scores,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # RAW PREDICTIONS
    # --------------------------------------------------------

    raw_predictions = (
        scores >= ERROR_THRESHOLD
    ).astype(np.int32)

    raw_error_sequence_count = int(
        np.sum(raw_predictions == 1)
    )

    # --------------------------------------------------------
    # TEMPORAL CONSISTENCY FILTER
    # --------------------------------------------------------

    predictions = apply_temporal_consistency(
        raw_predictions,
        scores,
    )

    error_sequence_count = int(
        np.sum(predictions == 1)
    )

    correct_count = int(
        np.sum(predictions == 0)
    )

    # --------------------------------------------------------
    # RAW ERROR REGIONS
    # --------------------------------------------------------

    raw_regions = []

    for index, prediction in enumerate(predictions):
        if prediction != 1:
            continue

        start_time, end_time = (
            sequence_times[index]
        )

        raw_regions.append(
            {
                "start": float(start_time),
                "end": float(end_time),
                "confidence": float(scores[index]),
                "sequence_start": index + 1,
                "sequence_end": index + 1,
            }
        )

    # --------------------------------------------------------
    # MERGE CONTINUOUS REGIONS
    # --------------------------------------------------------

    error_events = merge_error_regions(
        raw_regions
    )

    # Add event IDs.
    for event_id, event in enumerate(
        error_events,
        start=1,
    ):
        event["event_id"] = event_id

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    error_scores = scores[
        predictions == 1
    ]

    average_error_confidence = (
        float(np.mean(error_scores))
        if len(error_scores)
        else 0.0
    )

    # --------------------------------------------------------
    # OVERALL RESULT
    # --------------------------------------------------------

    overall_result = (
        "PROCEDURAL ERROR DETECTED"
        if error_events
        else "NO PROCEDURAL ERROR DETECTED"
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        "video_name": video_path.name,

        # Original video frame count.
        "original_frame_count": int(
            original_frame_count
        ),

        # Number of frames actually sampled for CNN-LSTM.
        "total_frames": int(
            len(frames)
        ),

        "video_duration": float(
            duration
        ),

        "total_sequences": int(
            len(sequences)
        ),

        "correct_count": int(
            correct_count
        ),

        # Temporal sequences classified as errors.
        "error_sequence_count": int(
            error_sequence_count
        ),

        # Raw model error count before temporal filtering.
        "raw_error_sequence_count": int(
            raw_error_sequence_count
        ),

        # Number of merged continuous events.
        "error_event_count": int(
            len(error_events)
        ),

        "average_error_confidence": float(
            average_error_confidence
        ),

        "result": overall_result,

        "error_events": error_events,

        # Backward compatibility with older app code.
        "error_regions": error_events,

        "scores": scores.tolist(),

        "predictions": predictions.tolist(),

        "raw_predictions": raw_predictions.tolist(),

        "sequence_times": [
            [
                float(start),
                float(end),
            ]
            for start, end in sequence_times
        ],
    }


# ============================================================
# TERMINAL MODE
# ============================================================

def main():
    print("=" * 60)
    print("PROCEVISION AI")
    print("CNN-LSTM V3 REAL VIDEO PREDICTION")
    print("=" * 60)

    video_input = input(
        "\nEnter video path: "
    ).strip()

    if not video_input:
        print("ERROR: No video path entered.")
        return

    video_path = Path(video_input)

    if not video_path.is_absolute():
        video_path = (
            PROJECT_ROOT
            / video_path
        )

    try:
        result = analyze_video(
            video_path.resolve()
        )

        print("\nVIDEO ANALYSIS SUMMARY")
        print("-" * 60)

        print(
            f"Original video frames : "
            f"{result['original_frame_count']}"
        )

        print(
            f"Sampled frames        : "
            f"{result['total_frames']}"
        )

        print(
            f"Temporal sequences     : "
            f"{result['total_sequences']}"
        )

        print(
            f"Correct sequences      : "
            f"{result['correct_count']}"
        )

        print(
            f"Error sequences        : "
            f"{result['error_sequence_count']}"
        )

        print(
            f"Continuous error events: "
            f"{result['error_event_count']}"
        )

        print("\nERROR EVENTS")
        print("-" * 60)

        if result["error_events"]:
            for event in result["error_events"]:
                print(
                    f"Event {event['event_id']}: "
                    f"{event['start']:.2f}s - "
                    f"{event['end']:.2f}s | "
                    f"Confidence: "
                    f"{event['confidence'] * 100:.2f}%"
                )
        else:
            print(
                "No procedural error events detected."
            )

        print("\nOVERALL RESULT")
        print("-" * 60)
        print(result["result"])

    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()