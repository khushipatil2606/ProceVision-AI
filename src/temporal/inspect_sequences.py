import numpy as np
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sequences"
)


# ============================================================
# INSPECT ONE CLASS
# ============================================================

def inspect_class(class_name):

    folder = SEQUENCE_DIR / class_name

    files = sorted(
        folder.glob("*.npz")
    )

    print("\n==========================================")
    print(f"{class_name.upper()} SEQUENCES")
    print("==========================================")

    print(
        f"Files found : {len(files)}"
    )

    valid = 0

    for file in files:

        try:

            data = np.load(file)

            frames = data["frames"]

            label = data["label"]

            expected_shape = (
                8,
                128,
                128,
                3
            )

            shape_ok = (
                frames.shape
                == expected_shape
            )

            if shape_ok:

                valid += 1

            print(
                f"{file.name:<35}"
                f"shape={frames.shape} "
                f"label={label}"
            )

        except Exception as e:

            print(
                f"ERROR: {file.name}"
            )

            print(e)

    print(
        f"\nValid sequences: "
        f"{valid}/{len(files)}"
    )

    return len(files), valid


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI - SEQUENCE INSPECTOR")
    print("==========================================")

    correct_total, correct_valid = (
        inspect_class("correct")
    )

    error_total, error_valid = (
        inspect_class("error")
    )

    print("\n==========================================")
    print("SEQUENCE DATASET SUMMARY")
    print("==========================================")

    print(
        f"Correct sequences : "
        f"{correct_total}"
    )

    print(
        f"Correct valid     : "
        f"{correct_valid}"
    )

    print(
        f"Error sequences   : "
        f"{error_total}"
    )

    print(
        f"Error valid       : "
        f"{error_valid}"
    )

    print(
        f"Total sequences   : "
        f"{correct_total + error_total}"
    )

    print("\n==========================================")

    if (
        correct_total == correct_valid
        and
        error_total == error_valid
    ):

        print(
            "SEQUENCE DATASET CHECK PASSED."
        )

    else:

        print(
            "WARNING: Invalid sequences found."
        )

    print("==========================================\n")


if __name__ == "__main__":
    main()