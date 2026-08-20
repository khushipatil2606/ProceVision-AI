import cv2
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Error video
VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scrambled_eggs"
    / "16_2_360p.mp4"
)

# Output directory for error frames
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "error_frames"
    / "16_2"
)


# Error interval from annotation
ERROR_START = 992.574
ERROR_END = 1071.976

# Save one frame every 2 seconds
INTERVAL_SECONDS = 2


def extract_error_frames(
    video_path,
    output_dir,
    start_time,
    end_time,
    interval_seconds
):

    if not video_path.exists():
        print(f"ERROR: Video not found:\n{video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        print("ERROR: Could not determine FPS.")
        cap.release()
        return

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_interval = int(interval_seconds * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_number = start_frame
    saved_count = 0

    print("\n======================================")
    print("   PROCEVISION AI - ERROR FRAMES")
    print("======================================")
    print(f"Video       : {video_path.name}")
    print(f"Error start : {start_time:.3f} seconds")
    print(f"Error end   : {end_time:.3f} seconds")
    print(f"Interval    : Every {interval_seconds} seconds")
    print(f"Output      : {output_dir}")
    print("======================================\n")

    while frame_number <= end_frame:

        success, frame = cap.read()

        if not success:
            break

        if (frame_number - start_frame) % frame_interval == 0:

            timestamp = frame_number / fps

            filename = (
                f"error_frame_{saved_count:03d}_"
                f"{timestamp:.2f}s.jpg"
            )

            output_path = output_dir / filename

            cv2.imwrite(str(output_path), frame)

            print(f"Saved: {filename}")

            saved_count += 1

        frame_number += 1

    cap.release()

    print("\n======================================")
    print("ERROR FRAME EXTRACTION COMPLETE")
    print("======================================")
    print(f"Frames saved : {saved_count}")
    print(f"Saved to     : {output_dir}")
    print("======================================\n")


if __name__ == "__main__":

    extract_error_frames(
        VIDEO_PATH,
        OUTPUT_DIR,
        ERROR_START,
        ERROR_END,
        INTERVAL_SECONDS
    )