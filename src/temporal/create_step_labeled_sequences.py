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
    / "step_labeled_sequences"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 128
SEQUENCE_LENGTH = 8
FRAME_INTERVAL = 1.0

# The known erroneous step in recording 16_2
ERROR_RECORDING = "16_2"
ERROR_STEP_ID = 170


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
# EXTRACT FRAMES FROM STEP
# ============================================================

def extract_step_frames(
    video_path,
    start_time,
    end_time
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"ERROR: Could not open:\n"
            f"{video_path}"
        )

        return [], []

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

    cap.release()

    return frames, timestamps


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    frames,
    timestamps
):

    sequences = []
    sequence_metadata = []

    if len(frames) < SEQUENCE_LENGTH:

        return (
            sequences,
            sequence_metadata
        )

    # Sliding window
    for start in range(
        0,
        len(frames) - SEQUENCE_LENGTH + 1
    ):

        end = (
            start
            + SEQUENCE_LENGTH
        )

        sequence = np.array(
            frames[start:end],
            dtype=np.float32
        )

        sequences.append(
            sequence
        )

        sequence_metadata.append({
            "start_time": timestamps[start],
            "end_time": timestamps[end - 1]
        })

    return (
        sequences,
        sequence_metadata
    )


# ============================================================
# SAVE SEQUENCE
# ============================================================

def save_sequence(
    sequence,
    output_dir,
    recording_id,
    step_id,
    sequence_number,
    label,
    start_time,
    end_time
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"{recording_id}"
        f"_step_{step_id}"
        f"_seq_{sequence_number:04d}"
        f"_{label}.npz"
    )

    output_path = (
        output_dir
        / filename
    )

    np.savez_compressed(
        output_path,
        frames=sequence,
        label=label,
        recording_id=recording_id,
        step_id=step_id,
        start_time=start_time,
        end_time=end_time
    )


# ============================================================
# PROCESS RECORDING
# ============================================================

def process_recording(
    recording,
    correct_dir,
    error_dir
):

    recording_id = recording[
        "recording_id"
    ]

    video_path = Path(
        recording["video"]
    )

    print("\n==========================================")
    print(
        f"PROCESSING RECORDING: "
        f"{recording_id}"
    )
    print("==========================================")

    print(
        f"Video : {video_path.name}"
    )

    total_sequences = 0
    correct_sequences = 0
    error_sequences = 0

    for step_index, step in enumerate(
        recording["steps"],
        start=1
    ):

        step_id = step.get(
            "step_id"
        )

        start_time = step.get(
            "start_time"
        )

        end_time = step.get(
            "end_time"
        )

        # ----------------------------------------------------
        # Determine label
        # ----------------------------------------------------

        if (
            recording_id == ERROR_RECORDING
            and
            step_id == ERROR_STEP_ID
        ):

            label = 1
            label_name = "error"
            output_dir = error_dir

        else:

            label = 0
            label_name = "correct"
            output_dir = correct_dir

        print(
            f"\nStep {step_index:02d} "
            f"(ID {step_id})"
        )

        print(
            f"Time : "
            f"{start_time:.2f}s - "
            f"{end_time:.2f}s"
        )

        print(
            f"Label: "
            f"{label_name.upper()}"
        )

        # ----------------------------------------------------
        # Extract frames
        # ----------------------------------------------------

        frames, timestamps = extract_step_frames(
            video_path,
            start_time,
            end_time
        )

        print(
            f"Frames: {len(frames)}"
        )

        # ----------------------------------------------------
        # Create sequences
        # ----------------------------------------------------

        sequences, sequence_metadata = (
            create_sequences(
                frames,
                timestamps
            )
        )

        print(
            f"Sequences: {len(sequences)}"
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        for sequence_number, sequence in enumerate(
            sequences
        ):

            meta = sequence_metadata[
                sequence_number
            ]

            save_sequence(
                sequence,
                output_dir,
                recording_id,
                step_id,
                sequence_number,
                label,
                meta["start_time"],
                meta["end_time"]
            )

        total_sequences += len(
            sequences
        )

        if label == 0:

            correct_sequences += len(
                sequences
            )

        else:

            error_sequences += len(
                sequences
            )

    return (
        total_sequences,
        correct_sequences,
        error_sequences
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" STEP-LABELED SEQUENCE GENERATOR")
    print("==========================================")

    print(
        "\nLoading metadata..."
    )

    metadata = load_metadata()

    # --------------------------------------------------------
    # Clean old output
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        import shutil

        shutil.rmtree(
            OUTPUT_DIR
        )

    correct_dir = (
        OUTPUT_DIR
        / "correct"
    )

    error_dir = (
        OUTPUT_DIR
        / "error"
    )

    correct_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    error_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find required recordings
    # --------------------------------------------------------

    recordings_to_process = []

    for recording in metadata:

        if recording[
            "recording_id"
        ] in [
            "16_1",
            "16_2"
        ]:

            recordings_to_process.append(
                recording
            )

    print(
        f"Recordings selected: "
        f"{len(recordings_to_process)}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    total = 0
    correct = 0
    error = 0

    for recording in recordings_to_process:

        result = process_recording(
            recording,
            correct_dir,
            error_dir
        )

        total += result[0]
        correct += result[1]
        error += result[2]

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n==========================================")
    print("STEP-LABELED DATASET COMPLETE")
    print("==========================================")

    print(
        f"Total sequences   : {total}"
    )

    print(
        f"Correct sequences : {correct}"
    )

    print(
        f"Error sequences   : {error}"
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_DIR}"
    )

    print("\n==========================================")
    print("IMPORTANT LABELING RULE")
    print("==========================================")

    print(
        "16_1 → all steps labeled CORRECT"
    )

    print(
        "16_2 → only Step ID 170 labeled ERROR"
    )

    print(
        "16_2 → all other steps labeled CORRECT"
    )

    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()