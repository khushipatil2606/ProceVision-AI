from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORRECT_SOURCE = PROJECT_ROOT / "data" / "processed" / "frames"
ERROR_SOURCE = PROJECT_ROOT / "data" / "processed" / "error_frames" / "16_2"

CORRECT_DEST = PROJECT_ROOT / "data" / "processed" / "dataset" / "correct"
ERROR_DEST = PROJECT_ROOT / "data" / "processed" / "dataset" / "error"


def copy_frames(source, destination):
    if not source.exists():
        print(f"ERROR: Source folder not found:\n{source}")
        return 0

    destination.mkdir(parents=True, exist_ok=True)

    files = list(source.glob("*.jpg"))

    print(f"\nSource      : {source}")
    print(f"Destination : {destination}")
    print(f"Frames found: {len(files)}")

    copied = 0

    for file in files:
        shutil.copy2(file, destination / file.name)
        copied += 1

    return copied


def main():

    print("\n==========================================")
    print("   PROCEVISION AI - DATASET ORGANIZER")
    print("==========================================")

    correct_count = copy_frames(
        CORRECT_SOURCE,
        CORRECT_DEST
    )

    error_count = copy_frames(
        ERROR_SOURCE,
        ERROR_DEST
    )

    print("\n==========================================")
    print("DATASET ORGANIZATION COMPLETE")
    print("==========================================")
    print(f"Correct frames : {correct_count}")
    print(f"Error frames   : {error_count}")
    print("==========================================\n")


if __name__ == "__main__":
    main()