from pathlib import Path
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"

CORRECT_DIR = DATASET_DIR / "correct"
ERROR_DIR = DATASET_DIR / "error"


def inspect_folder(folder, label):
    images = list(folder.glob("*.jpg"))

    print(f"\n{label.upper()}")
    print("-" * 40)
    print(f"Folder : {folder}")
    print(f"Images : {len(images)}")

    valid = 0
    invalid = 0

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()
            valid += 1
        except Exception:
            invalid += 1
            print(f"Invalid image: {image_path.name}")

    print(f"Valid   : {valid}")
    print(f"Invalid : {invalid}")

    return len(images), valid, invalid


def main():

    print("\n==========================================")
    print("   PROCEVISION AI - DATASET INSPECTOR")
    print("==========================================")

    correct_total, correct_valid, correct_invalid = inspect_folder(
        CORRECT_DIR,
        "Correct"
    )

    error_total, error_valid, error_invalid = inspect_folder(
        ERROR_DIR,
        "Error"
    )

    total = correct_total + error_total
    valid = correct_valid + error_valid
    invalid = correct_invalid + error_invalid

    print("\n==========================================")
    print("DATASET SUMMARY")
    print("==========================================")
    print(f"Total images : {total}")
    print(f"Valid images : {valid}")
    print(f"Invalid      : {invalid}")
    print(f"Correct      : {correct_total}")
    print(f"Error        : {error_total}")
    print("==========================================")

    if invalid == 0:
        print("\nDataset check PASSED.")
        print("All images are valid.")
    else:
        print("\nDataset check FAILED.")
        print("Some images need to be removed/fixed.")

    print("==========================================\n")


if __name__ == "__main__":
    main()