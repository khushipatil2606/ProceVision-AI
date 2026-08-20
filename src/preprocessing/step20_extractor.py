import cv2
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# VIDEO PATHS
# ============================================================

VIDEO_16_1 = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scrambled_eggs"
    / "16_1_360p.mp4"
)

VIDEO_16_2 = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scrambled_eggs"
    / "16_2_360p.mp4"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

CORRECT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20"
    / "correct"
)

ERROR_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20"
    / "error"
)


# ============================================================
# STEP 20 TIMESTAMPS
# ============================================================

# These values come from the annotations.
#
# IMPORTANT:
# Replace the 16_1 values below with the exact values
# printed by your step_comparator.py output.

CORRECT_START = 916.486
CORRECT_END = 979.587

ERROR_START = 992.574
ERROR_END = 1071.976


# ============================================================
# FRAME INTERVAL
# ============================================================

INTERVAL_SECONDS = 2


# ============================================================
# EXTRACT FRAMES
# ============================================================

def extract_frames(
    video_path,
    output_dir,
    start_time,
    end_time,
    label
):

    if not video_path.exists():

        print("\nERROR: Video not found:")
        print(video_path)

        return

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print("\nERROR: Could not open video:")
        print(video_path)

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        print("\nERROR: Invalid FPS.")

        cap.release()

        return

    start_frame = int(
        start_time * fps
    )

    end_frame = int(
        end_time * fps
    )

    frame_interval = max(
        1,
        int(INTERVAL_SECONDS * fps)
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    frame_number = start_frame

    saved_count = 0

    print("\n==========================================")
    print(f"Extracting {label} Step 20")
    print("==========================================")
    print(f"Video       : {video_path.name}")
    print(f"Start       : {start_time:.3f}s")
    print(f"End         : {end_time:.3f}s")
    print(f"Interval    : {INTERVAL_SECONDS}s")
    print(f"Output      : {output_dir}")
    print("==========================================")

    while frame_number <= end_frame:

        success, frame = cap.read()

        if not success:
            break

        if (
            (frame_number - start_frame)
            % frame_interval
            == 0
        ):

            timestamp = frame_number / fps

            filename = (
                f"step20_{label}_"
                f"{saved_count:03d}_"
                f"{timestamp:.2f}s.jpg"
            )

            output_path = (
                output_dir / filename
            )

            cv2.imwrite(
                str(output_path),
                frame
            )

            saved_count += 1

        frame_number += 1

    cap.release()

    print(
        f"\nFrames saved: {saved_count}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - STEP 20 EXTRACTOR")
    print("==========================================")

    # Correct recording
    extract_frames(
        VIDEO_16_1,
        CORRECT_OUTPUT,
        CORRECT_START,
        CORRECT_END,
        "correct"
    )

    # Error recording
    extract_frames(
        VIDEO_16_2,
        ERROR_OUTPUT,
        ERROR_START,
        ERROR_END,
        "error"
    )

    print("\n==========================================")
    print("STEP 20 EXTRACTION COMPLETE")
    print("==========================================\n")


if __name__ == "__main__":
    main()