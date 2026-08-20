import numpy as np
import shutil
import random

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
    / "step_labeled_balanced"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_ERROR_COUNT = 1000

RANDOM_SEED = 42


# ============================================================
# COPY CORRECT DATA
# ============================================================

def copy_correct():

    source = (
        SOURCE_DIR
        / "correct"
    )

    destination = (
        OUTPUT_DIR
        / "correct"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    files = list(
        source.glob("*.npz")
    )

    for file in files:

        shutil.copy2(
            file,
            destination / file.name
        )

    return len(files)


# ============================================================
# AUGMENT ERROR SEQUENCE
# ============================================================

def augment_sequence(frames):

    augmented = frames.copy()

    # Small brightness variation
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

    # Small Gaussian noise
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
# CREATE ERROR DATA
# ============================================================

def create_error_data():

    source = (
        SOURCE_DIR
        / "error"
    )

    destination = (
        OUTPUT_DIR
        / "error"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    original_files = list(
        source.glob("*.npz")
    )

    print(
        f"\nOriginal error sequences : "
        f"{len(original_files)}"
    )

    # --------------------------------------------------------
    # Copy originals
    # --------------------------------------------------------

    for file in original_files:

        shutil.copy2(
            file,
            destination / file.name
        )

    # --------------------------------------------------------
    # Generate augmented samples
    # --------------------------------------------------------

    generated = 0

    while (
        len(original_files)
        + generated
        < TARGET_ERROR_COUNT
    ):

        source_file = random.choice(
            original_files
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
            destination
            / filename
        )

        np.savez_compressed(
            output_file,
            frames=augmented,
            label=1,
            source_file=source_file.name
        )

        generated += 1

    return (
        len(original_files),
        generated
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" ERROR SEQUENCE BALANCER")
    print("==========================================")

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Remove old output
    # --------------------------------------------------------

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    # --------------------------------------------------------
    # Correct
    # --------------------------------------------------------

    correct_count = copy_correct()

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    original_error, generated_error = (
        create_error_data()
    )

    final_error_count = (
        original_error
        + generated_error
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n==========================================")
    print("BALANCED DATASET COMPLETE")
    print("==========================================")

    print(
        f"Correct sequences : "
        f"{correct_count}"
    )

    print(
        f"Original errors   : "
        f"{original_error}"
    )

    print(
        f"Augmented errors  : "
        f"{generated_error}"
    )

    print(
        f"Total errors      : "
        f"{final_error_count}"
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_DIR}"
    )

    print("\n==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()