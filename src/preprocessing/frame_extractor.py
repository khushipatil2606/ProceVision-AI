import cv2
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input video
VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scrambled_eggs"
    / "16_1_360p.mp4"
)

# Output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "correct_frames"
    / "16_1"
)


# Extract one frame every N seconds
INTERVAL_SECONDS = 2


def extract_frames(video_path, output_dir, interval_seconds):
    if not video_path.exists():
        print(f"ERROR: Video not found:\n{video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("ERROR: Could not determine FPS.")
        cap.release()
        return

    duration = total_frames / fps

    frame_interval = int(fps * interval_seconds)

    print("\n========== FRAME EXTRACTION ==========")
    print(f"Video       : {video_path.name}")
    print(f"FPS         : {fps:.2f}")
    print(f"Duration    : {duration / 60:.2f} minutes")
    print(f"Interval    : Every {interval_seconds} seconds")
    print(f"Output      : {output_dir}")
    print("======================================\n")

    frame_number = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % frame_interval == 0:
            timestamp = frame_number / fps

            filename = (
                f"frame_{saved_count:05d}_"
                f"{timestamp:.2f}s.jpg"
            )

            output_path = output_dir / filename

            cv2.imwrite(str(output_path), frame)

            saved_count += 1

        frame_number += 1

    cap.release()

    print("\n========== EXTRACTION COMPLETE ==========")
    print(f"Frames processed : {frame_number}")
    print(f"Frames saved     : {saved_count}")
    print(f"Saved to         : {output_dir}")
    print("=========================================\n")


if __name__ == "__main__":
    extract_frames(
        VIDEO_PATH,
        OUTPUT_DIR,
        INTERVAL_SECONDS
    )