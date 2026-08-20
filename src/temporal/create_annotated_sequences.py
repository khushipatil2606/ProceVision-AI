import json
import cv2
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_metadata"
    / "recordings.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "annotated_sequences"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 128

SEQUENCE_LENGTH = 8

FRAME_INTERVAL = 2

MIN_FRAMES_PER_SEQUENCE = 8


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# RESIZE FRAME
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
# EXTRACT FRAMES FROM STEP
# ============================================================

def extract_step_frames(
    video,
    start_time,
    end_time
):

    frames = []

    current_time = start_time

    while current_time <= end_time:

        video.set(
            cv2.CAP_PROP_POS_MSEC,
            current_time * 1000
        )

        success, frame = video.read()

        if not success:

            break

        processed = preprocess_frame(
            frame
        )

        frames.append(
            processed
        )

        current_time += FRAME_INTERVAL

    return frames


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    frames
):

    sequences = []

    if len(frames) < MIN_FRAMES_PER_SEQUENCE:

        return sequences

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

        if len(sequence) == SEQUENCE_LENGTH:

            sequences.append(
                np.array(
                    sequence,
                    dtype=np.float32
                )
            )

    return sequences


# ============================================================
# PROCESS RECORDING
# ============================================================

def process_recording(
    recording
):

    recording_id = recording[
        "recording_id"
    ]

    label = recording[
        "label"
    ]

    video_path = recording[
        "video"
    ]

    steps = recording.get(
        "steps",
        []
    )

    print("\n==========================================")
    print(
        f"Recording : {recording_id}"
    )
    print(
        f"Label     : {label}"
    )
    print(
        f"Steps     : {len(steps)}"
    )
    print("==========================================")

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    video = cv2.VideoCapture(
        video_path
    )

    if not video.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return 0

    total_sequences = 0

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    label_dir = (
        OUTPUT_DIR
        / label
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process each step
    # --------------------------------------------------------

    for step_index, step in enumerate(
        steps
    ):

        step_id = step.get(
            "step_id"
        )

        description = step.get(
            "description",
            ""
        )

        start_time = step[
            "start_time"
        ]

        end_time = step[
            "end_time"
        ]

        print(
            f"\nStep {step_index + 1}: "
            f"ID={step_id}"
        )

        print(
            f"Time: "
            f"{start_time:.2f}s - "
            f"{end_time:.2f}s"
        )

        # ----------------------------------------------------
        # Extract frames
        # ----------------------------------------------------

        frames = extract_step_frames(
            video,
            start_time,
            end_time
        )

        print(
            f"Frames extracted: "
            f"{len(frames)}"
        )

        # ----------------------------------------------------
        # Create sequences
        # ----------------------------------------------------

        sequences = create_sequences(
            frames
        )

        print(
            f"Sequences created: "
            f"{len(sequences)}"
        )

        # ----------------------------------------------------
        # Save sequences
        # ----------------------------------------------------

        for sequence_index, sequence in enumerate(
            sequences
        ):

            filename = (
                f"{recording_id}_"
                f"step_{step_id}_"
                f"seq_{sequence_index:03d}.npz"
            )

            output_file = (
                label_dir
                / filename
            )

            np.savez_compressed(
                output_file,
                frames=sequence,
                label=(
                    1
                    if label == "error"
                    else 0
                ),
                recording_id=recording_id,
                step_id=step_id
            )

            total_sequences += 1

    video.release()

    print(
        f"\nTotal sequences from "
        f"{recording_id}: "
        f"{total_sequences}"
    )

    return total_sequences


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" ANNOTATED TEMPORAL SEQUENCE GENERATOR")
    print("==========================================")

    print(
        f"\nSequence length : "
        f"{SEQUENCE_LENGTH} frames"
    )

    print(
        f"Frame interval  : "
        f"{FRAME_INTERVAL} seconds"
    )

    print(
        f"Image size      : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    recordings = load_metadata()

    print(
        f"\nRecordings loaded: "
        f"{len(recordings)}"
    )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    (
        OUTPUT_DIR / "correct"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        OUTPUT_DIR / "error"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process recordings
    # --------------------------------------------------------

    total_correct = 0
    total_error = 0

    for recording in recordings:

        count = process_recording(
            recording
        )

        if recording["label"] == "correct":

            total_correct += count

        else:

            total_error += count

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n==========================================")
    print("ANNOTATED SEQUENCE GENERATION COMPLETE")
    print("==========================================")

    print(
        f"Correct sequences : "
        f"{total_correct}"
    )

    print(
        f"Error sequences   : "
        f"{total_error}"
    )

    print(
        f"Total sequences   : "
        f"{total_correct + total_error}"
    )

    print("\nSaved to:")

    print(
        OUTPUT_DIR
    )

    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()