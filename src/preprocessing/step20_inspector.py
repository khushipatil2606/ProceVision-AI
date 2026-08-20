from pathlib import Path
from PIL import Image


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


CORRECT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20"
    / "correct"
)

ERROR_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20"
    / "error"
)


# ============================================================
# INSPECT FOLDER
# ============================================================

def inspect_folder(folder, label):

    print("\n==========================================")
    print(f"{label.upper()} STEP 20")
    print("==========================================")

    if not folder.exists():

        print("Folder not found:")
        print(folder)

        return 0, 0

    images = list(folder.glob("*.jpg"))

    valid = 0
    invalid = 0

    for image_path in images:

        try:

            with Image.open(image_path) as img:

                img.verify()

            valid += 1

        except Exception:

            invalid += 1

            print(
                f"Invalid image: {image_path.name}"
            )

    print(f"Folder  : {folder}")
    print(f"Images  : {len(images)}")
    print(f"Valid   : {valid}")
    print(f"Invalid : {invalid}")

    return len(images), valid


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - STEP 20 INSPECTOR")
    print("==========================================")

    correct_total, correct_valid = inspect_folder(
        CORRECT_DIR,
        "Correct"
    )

    error_total, error_valid = inspect_folder(
        ERROR_DIR,
        "Error"
    )

    total = correct_total + error_total
    valid = correct_valid + error_valid

    print("\n==========================================")
    print("STEP 20 DATASET SUMMARY")
    print("==========================================")

    print(f"Total images : {total}")
    print(f"Valid images : {valid}")
    print(f"Invalid      : {total - valid}")
    print(f"Correct      : {correct_total}")
    print(f"Error        : {error_total}")

    print("\n==========================================")

    if total == valid:

        print("STEP 20 DATASET CHECK PASSED.")

    else:

        print("WARNING: Invalid images found.")

    print("==========================================\n")


if __name__ == "__main__":
    main()