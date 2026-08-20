from pathlib import Path
import shutil
import random


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "balanced_dataset"

CORRECT_SOURCE = SOURCE_DIR / "correct"
ERROR_SOURCE = SOURCE_DIR / "error"

CORRECT_DEST = OUTPUT_DIR / "correct"
ERROR_DEST = OUTPUT_DIR / "error"


def copy_images(source, destination, images):
    destination.mkdir(parents=True, exist_ok=True)

    for image in images:
        shutil.copy2(image, destination / image.name)


def main():

    print("\n==========================================")
    print("   PROCEVISION AI - DATASET BALANCER")
    print("==========================================")

    correct_images = list(CORRECT_SOURCE.glob("*.jpg"))
    error_images = list(ERROR_SOURCE.glob("*.jpg"))

    print(f"\nOriginal Correct images : {len(correct_images)}")
    print(f"Original Error images   : {len(error_images)}")

    if len(error_images) == 0:
        print("ERROR: No error images found.")
        return

    # Use the error class size as the target.
    target_size = len(error_images)

    random.seed(42)

    # Randomly select the same number of correct images
    # to create a balanced dataset.
    selected_correct = random.sample(
        correct_images,
        min(target_size, len(correct_images))
    )

    # Copy error images
    copy_images(
        ERROR_SOURCE,
        ERROR_DEST,
        error_images
    )

    # Copy selected correct images
    copy_images(
        CORRECT_SOURCE,
        CORRECT_DEST,
        selected_correct
    )

    print("\n==========================================")
    print("BALANCING COMPLETE")
    print("==========================================")
    print(f"Correct images : {len(selected_correct)}")
    print(f"Error images   : {len(error_images)}")
    print(f"Total images   : {len(selected_correct) + len(error_images)}")
    print(f"Output         : {OUTPUT_DIR}")
    print("==========================================\n")


if __name__ == "__main__":
    main()