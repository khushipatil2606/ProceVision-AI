from pathlib import Path
import shutil
import random


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# SOURCE
# ============================================================

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20_balanced"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "step20_split"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


# ============================================================
# SPLIT ONE CLASS
# ============================================================

def split_class(class_name):

    source = SOURCE_DIR / class_name

    if not source.exists():
        print(f"ERROR: Folder not found: {source}")
        return

    images = list(source.glob("*.jpg"))

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    # --------------------------------------------------------
    # CREATE DIRECTORIES
    # --------------------------------------------------------

    train_dir = OUTPUT_DIR / "train" / class_name
    val_dir = OUTPUT_DIR / "validation" / class_name
    test_dir = OUTPUT_DIR / "test" / class_name

    train_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    val_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # COPY FILES
    # --------------------------------------------------------

    for image in train_images:

        shutil.copy2(
            image,
            train_dir / image.name
        )

    for image in val_images:

        shutil.copy2(
            image,
            val_dir / image.name
        )

    for image in test_images:

        shutil.copy2(
            image,
            test_dir / image.name
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(f"\nClass: {class_name}")
    print(f"Total      : {total}")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test       : {len(test_images)}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - STEP 20 SPLITTER")
    print("==========================================")

    print(f"\nSource:")
    print(SOURCE_DIR)

    print(f"\nOutput:")
    print(OUTPUT_DIR)

    # Process both classes
    split_class("correct")
    split_class("error")

    print("\n==========================================")
    print("STEP 20 DATASET SPLIT COMPLETE")
    print("==========================================")

    print("\nDataset structure:")

    print("""
step20_split/
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
""")

    print("==========================================\n")


if __name__ == "__main__":
    main()