from pathlib import Path
import shutil
import random


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# SOURCE DIRECTORIES
# ============================================================

CORRECT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20"
    / "correct"
)

ERROR_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20"
    / "error"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20_balanced"
)


CORRECT_OUTPUT = OUTPUT_DIR / "correct"
ERROR_OUTPUT = OUTPUT_DIR / "error"


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)


# ============================================================
# COPY IMAGES
# ============================================================

def copy_images(source, destination, number):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    images = list(
        source.glob("*.jpg")
    )

    random.shuffle(images)

    selected_images = images[:number]

    for image in selected_images:

        shutil.copy2(
            image,
            destination / image.name
        )

    return selected_images


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - STEP 20 BALANCER")
    print("==========================================")

    if not CORRECT_SOURCE.exists():

        print("ERROR: Correct folder not found.")
        print(CORRECT_SOURCE)
        return

    if not ERROR_SOURCE.exists():

        print("ERROR: Error folder not found.")
        print(ERROR_SOURCE)
        return


    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    correct_images = list(
        CORRECT_SOURCE.glob("*.jpg")
    )

    error_images = list(
        ERROR_SOURCE.glob("*.jpg")
    )


    print("\nOriginal dataset:")
    print(f"Correct : {len(correct_images)}")
    print(f"Error   : {len(error_images)}")


    # --------------------------------------------------------
    # BALANCE USING SMALLER CLASS
    # --------------------------------------------------------

    target_count = min(
        len(correct_images),
        len(error_images)
    )


    print(
        f"\nTarget per class : {target_count}"
    )


    # --------------------------------------------------------
    # COPY CORRECT
    # --------------------------------------------------------

    selected_correct = copy_images(
        CORRECT_SOURCE,
        CORRECT_OUTPUT,
        target_count
    )


    # --------------------------------------------------------
    # COPY ERROR
    # --------------------------------------------------------

    selected_error = copy_images(
        ERROR_SOURCE,
        ERROR_OUTPUT,
        target_count
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n==========================================")
    print("BALANCING COMPLETE")
    print("==========================================")

    print(
        f"Correct images : {len(selected_correct)}"
    )

    print(
        f"Error images   : {len(selected_error)}"
    )

    print(
        f"Total images   : "
        f"{len(selected_correct) + len(selected_error)}"
    )

    print(
        f"\nOutput : {OUTPUT_DIR}"
    )

    print("==========================================\n")


if __name__ == "__main__":
    main()