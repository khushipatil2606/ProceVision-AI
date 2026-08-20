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
    / "annotated_sequences"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "annotated_split"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# SPLIT CLASS
# ============================================================

def split_class(class_name):

    source = SOURCE_DIR / class_name

    files = sorted(
        source.glob("*.npz")
    )

    random.shuffle(files)

    total = len(files)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    # Remaining files go to test
    test_count = (
        total
        - train_count
        - val_count
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
    print(f"Class: {class_name}")
    print("==========================================")

    print(
        f"Total : {total}"
    )

    print(
        f"Train : {len(train_files)}"
    )

    print(
        f"Val   : {len(val_files)}"
    )

    print(
        f"Test  : {len(test_files)}"
    )

    # --------------------------------------------------------
    # Copy files
    # --------------------------------------------------------

    splits = {
        "train": train_files,
        "validation": val_files,
        "test": test_files
    }

    for split_name, split_files in splits.items():

        destination = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for file in split_files:

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
    print(" ANNOTATED TEMPORAL DATASET SPLITTER")
    print("==========================================")

    random.seed(
        RANDOM_SEED
    )

    print(
        f"\nTrain ratio : {TRAIN_RATIO}"
    )

    print(
        f"Validation  : {VAL_RATIO}"
    )

    print(
        f"Test        : {TEST_RATIO}"
    )

    # --------------------------------------------------------
    # Clean old split
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    # --------------------------------------------------------
    # Split both classes
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
    print("ANNOTATED DATASET SPLIT COMPLETE")
    print("==========================================")

    print("\nOutput:")

    print(
        OUTPUT_DIR
    )

    print("\nStructure:")

    print(
        """
annotated_split/
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