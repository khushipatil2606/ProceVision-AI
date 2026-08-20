import cv2
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Video location
VIDEO_PATH = PROJECT_ROOT / "data" / "raw" / "scrambled_eggs" / "16_1_360p.mp4"


def inspect_video(video_path):
    print("\n========== PROCEVISION AI ==========")
    print("Video Inspector")
    print("====================================")

    # Check whether the video exists
    if not video_path.exists():
        print(f"\nERROR: Video not found!")
        print(f"Expected location:\n{video_path}")
        return

    print(f"\nVideo found: {video_path.name}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("\nERROR: Could not open the video.")
        return

    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate duration
    duration = frame_count / fps if fps > 0 else 0

    print("\n---------- VIDEO INFORMATION ----------")
    print(f"File name       : {video_path.name}")
    print(f"Resolution      : {width} x {height}")
    print(f"FPS             : {fps:.2f}")
    print(f"Total frames    : {frame_count}")
    print(f"Duration        : {duration:.2f} seconds")
    print(f"Duration        : {duration / 60:.2f} minutes")
    print("---------------------------------------")

    # Read the first frame
    success, frame = cap.read()

    if success:
        print("First frame     : Successfully read")
        print(f"Frame shape     : {frame.shape}")
    else:
        print("First frame     : Could not be read")

    cap.release()

    print("\nVideo inspection completed successfully.")
    print("====================================\n")


if __name__ == "__main__":
    inspect_video(VIDEO_PATH)