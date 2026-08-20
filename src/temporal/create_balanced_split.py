import random
import shutil
import numpy as np

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
    / "balanced_split"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

TARGET_TRAIN_ERRORS = 1000

RANDOM_SEED = 42


# ============================================================
# AUGMENT
# ============================================================

def augment_sequence(frames):

    augmented = frames.copy()

    # Brightness variation
    brightness = random.uniform(
        0.90,
        1.10
    )

    augmented *= brightness

    augmented = np.clip(
        augmented,
        0.0,
        1.0
    )

    # Small noise
    noise = np.random.normal(
        0,
        0.01,
        augmented.shape
    ).astype(
        np.float32
    )

    augmented += noise

    augmented = np.clip(
        augmented,
        0.0,
        1.0
    )

    return augmented.astype(
        np.float32
    )


# ============================================================
# COPY FILE
# ============================================================

def copy_file(
    source,
    destination
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" LEAKAGE-FREE BALANCED SPLITTER")
    print("==========================================")

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Remove previous output
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    # --------------------------------------------------------
    # Load ORIGINAL files
    # --------------------------------------------------------

    correct_files = list(
        (
            SOURCE_DIR / "correct"
        ).glob("*.npz")
    )

    error_files = list(
        (
            SOURCE_DIR / "error"
        ).glob("*.npz")
    )

    random.shuffle(
        correct_files
    )

    random.shuffle(
        error_files
    )

    print(
        f"\nOriginal correct : "
        f"{len(correct_files)}"
    )

    print(
        f"Original error   : "
        f"{len(error_files)}"
    )

    # --------------------------------------------------------
    # Split function
    # --------------------------------------------------------

    def split_files(files):

        total = len(files)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        train = files[
            :train_count
        ]

        validation = files[
            train_count:
            train_count + val_count
        ]

        test = files[
            train_count + val_count:
        ]

        return (
            train,
            validation,
            test
        )

    correct_train, correct_val, correct_test = (
        split_files(
            correct_files
        )
    )

    error_train, error_val, error_test = (
        split_files(
            error_files
        )
    )

    # --------------------------------------------------------
    # Copy correct data
    # --------------------------------------------------------

    for split_name, files in [

        ("train", correct_train),
        ("validation", correct_val),
        ("test", correct_test)

    ]:

        for file in files:

            copy_file(
                file,
                OUTPUT_DIR
                / split_name
                / "correct"
                / file.name
            )

    # --------------------------------------------------------
    # Copy ORIGINAL error data
    # --------------------------------------------------------

    for split_name, files in [

        ("validation", error_val),
        ("test", error_test)

    ]:

        for file in files:

            copy_file(
                file,
                OUTPUT_DIR
                / split_name
                / "error"
                / file.name
            )

    # --------------------------------------------------------
    # Copy training errors
    # --------------------------------------------------------

    for file in error_train:

        copy_file(
            file,
            OUTPUT_DIR
            / "train"
            / "error"
            / file.name
        )

    # --------------------------------------------------------
    # Augment ONLY training errors
    # --------------------------------------------------------

    generated = 0

    while (
        len(error_train)
        + generated
        < TARGET_TRAIN_ERRORS
    ):

        source_file = random.choice(
            error_train
        )

        data = np.load(
            source_file,
            allow_pickle=True
        )

        frames = data[
            "frames"
        ]

        augmented = augment_sequence(
            frames
        )

        filename = (
            f"aug_error_"
            f"{generated:05d}.npz"
        )

        output_file = (
            OUTPUT_DIR
            / "train"
            / "error"
            / filename
        )

        np.savez_compressed(
            output_file,
            frames=augmented,
            label=1,
            source_file=source_file.name
        )

        generated += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n==========================================")
    print("DATASET SPLIT SUMMARY")
    print("==========================================")

    print("\nCORRECT")

    print(
        f"Train      : "
        f"{len(correct_train)}"
    )

    print(
        f"Validation : "
        f"{len(correct_val)}"
    )

    print(
        f"Test       : "
        f"{len(correct_test)}"
    )

    print("\nERROR")

    print(
        f"Original train : "
        f"{len(error_train)}"
    )

    print(
        f"Augmented train: "
        f"{generated}"
    )

    print(
        f"Total train    : "
        f"{len(error_train) + generated}"
    )

    print(
        f"Validation     : "
        f"{len(error_val)}"
    )

    print(
        f"Test           : "
        f"{len(error_test)}"
    )

    print("\n==========================================")
    print("LEAKAGE-FREE SPLIT COMPLETE")
    print("==========================================")

    print(
        f"\nOutput:\n"
        f"{OUTPUT_DIR}"
    )

    print(
        """
balanced_split/
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