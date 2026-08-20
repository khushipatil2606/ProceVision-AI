import cv2
import numpy as np

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# VIDEO PATHS
# ============================================================

CORRECT_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scrambled_eggs"
    / "16_1_360p.mp4"
)

ERROR_VIDEO = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scrambled_eggs"
    / "16_2_360p.mp4"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)


# ============================================================
# STEP 20 TIME RANGES
# ============================================================

CORRECT_START = 916.486
CORRECT_END = 979.587

ERROR_START = 992.574
ERROR_END = 1071.976


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

FRAME_INTERVAL = 2

SEQUENCE_LENGTH = 8


# ============================================================
# EXTRACT FRAMES FROM VIDEO
# ============================================================

def extract_frames(
    video_path,
    start_time,
    end_time
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        cap.release()

        raise RuntimeError(
            "Could not determine FPS."
        )

    start_frame = int(
        start_time * fps
    )

    end_frame = int(
        end_time * fps
    )

    frame_step = int(
        FRAME_INTERVAL * fps
    )

    frames = []

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    current_frame = start_frame

    while current_frame <= end_frame:

        success, frame = cap.read()

        if not success:
            break

        # Resize frame
        frame = cv2.resize(
            frame,
            IMAGE_SIZE
        )

        # Convert BGR → RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Normalize
        frame = (
            frame.astype(
                np.float32
            ) / 255.0
        )

        frames.append(frame)

        # Jump forward
        next_frame = (
            current_frame + frame_step
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            next_frame
        )

        current_frame = next_frame

    cap.release()

    return frames


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    frames,
    label
):

    sequences = []

    total_frames = len(frames)

    if total_frames < SEQUENCE_LENGTH:

        return sequences

    for start in range(
        0,
        total_frames - SEQUENCE_LENGTH + 1,
        SEQUENCE_LENGTH
    ):

        sequence = frames[
            start:
            start + SEQUENCE_LENGTH
        ]

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        sequences.append(
            (
                sequence,
                label
            )
        )

    return sequences


# ============================================================
# SAVE SEQUENCES
# ============================================================

def save_sequences(
    sequences,
    label
):

    label_dir = (
        OUTPUT_DIR / label
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, (
        sequence,
        sequence_label
    ) in enumerate(sequences):

        filename = (
            f"{label}_sequence_"
            f"{index:03d}.npz"
        )

        output_path = (
            label_dir / filename
        )

        np.savez_compressed(
            output_path,
            frames=sequence,
            label=sequence_label
        )

    return len(sequences)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI - SEQUENCE GENERATOR")
    print("==========================================")

    print(
        f"\nSequence length : "
        f"{SEQUENCE_LENGTH} frames"
    )

    print(
        f"Frame interval  : "
        f"{FRAME_INTERVAL} seconds"
    )


    # ========================================================
    # CORRECT
    # ========================================================

    print("\n==========================================")
    print("PROCESSING CORRECT VIDEO")
    print("==========================================")

    correct_frames = extract_frames(
        CORRECT_VIDEO,
        CORRECT_START,
        CORRECT_END
    )

    print(
        f"Frames extracted: "
        f"{len(correct_frames)}"
    )

    correct_sequences = create_sequences(
        correct_frames,
        0
    )

    print(
        f"Sequences created: "
        f"{len(correct_sequences)}"
    )

    correct_saved = save_sequences(
        correct_sequences,
        "correct"
    )


    # ========================================================
    # ERROR
    # ========================================================

    print("\n==========================================")
    print("PROCESSING ERROR VIDEO")
    print("==========================================")

    error_frames = extract_frames(
        ERROR_VIDEO,
        ERROR_START,
        ERROR_END
    )

    print(
        f"Frames extracted: "
        f"{len(error_frames)}"
    )

    error_sequences = create_sequences(
        error_frames,
        1
    )

    print(
        f"Sequences created: "
        f"{len(error_sequences)}"
    )

    error_saved = save_sequences(
        error_sequences,
        "error"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n==========================================")
    print("SEQUENCE GENERATION COMPLETE")
    print("==========================================")

    print(
        f"Correct sequences : "
        f"{correct_saved}"
    )

    print(
        f"Error sequences   : "
        f"{error_saved}"
    )

    print(
        f"Total sequences   : "
        f"{correct_saved + error_saved}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_DIR}"
    )

    print("==========================================\n")


if __name__ == "__main__":
    main()