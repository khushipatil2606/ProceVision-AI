import random
import shutil
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step_labeled_sequences"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step_labeled_split"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# SPLIT DATA
# ============================================================

def split_class(class_name):

    source_dir = SOURCE_DIR / class_name

    files = sorted(
        source_dir.glob("*.npz")
    )

    random.shuffle(files)

    total = len(files)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    train_files = files[
        :train_count
    ]

    val_files = files[
        train_count:
        train_count + val_count
    ]

    test_files = files[
        train_count + val_count:
    ]

    print("\n==========================================")
    print(f"CLASS: {class_name.upper()}")
    print("==========================================")

    print(
        f"Total      : {total}"
    )

    print(
        f"Train      : {len(train_files)}"
    )

    print(
        f"Validation : {len(val_files)}"
    )

    print(
        f"Test       : {len(test_files)}"
    )

    split_data = {
        "train": train_files,
        "validation": val_files,
        "test": test_files
    }

    for split_name, files_to_copy in split_data.items():

        destination = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for file in files_to_copy:

            shutil.copy2(
                file,
                destination / file.name
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" STEP-LABELED DATASET SPLITTER")
    print("==========================================")

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_DIR.exists():

        print(
            "\nERROR: Source dataset not found:"
        )

        print(
            SOURCE_DIR
        )

        return

    # --------------------------------------------------------
    # Remove old split
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    # --------------------------------------------------------
    # Split classes
    # --------------------------------------------------------

    split_class(
        "correct"
    )

    split_class(
        "error"
    )

    # --------------------------------------------------------
    # Final structure
    # --------------------------------------------------------

    print("\n==========================================")
    print("STEP-LABELED DATASET SPLIT COMPLETE")
    print("==========================================")

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )

    print(
        """
step_labeled_split/
│
├── train/
│   ├── correct/
│   └── error/
│
├── validation/
│   ├── correct/
│   └── error/
│
└── test/
    ├── correct/
    └── error/
"""
    )

    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()