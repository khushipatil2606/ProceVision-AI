import numpy as np
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "annotated_sequences"
)


# ============================================================
# EXPECTED SHAPE
# ============================================================

EXPECTED_SHAPE = (
    8,
    128,
    128,
    3
)


# ============================================================
# INSPECT CLASS
# ============================================================

def inspect_class(label):

    folder = SEQUENCE_DIR / label

    print("\n==========================================")
    print(f"{label.upper()} SEQUENCES")
    print("==========================================")

    if not folder.exists():

        print("Folder does not exist.")

        return 0, 0

    files = sorted(
        folder.glob("*.npz")
    )

    print(
        f"Files found : {len(files)}"
    )

    valid = 0
    invalid = 0

    for file in files:

        try:

            data = np.load(
                file,
                allow_pickle=True
            )

            frames = data["frames"]

            label_value = data["label"]

            recording_id = data.get(
                "recording_id",
                "unknown"
            )

            step_id = data.get(
                "step_id",
                "unknown"
            )

            # ------------------------------------------------
            # Check shape
            # ------------------------------------------------

            shape_valid = (
                frames.shape
                == EXPECTED_SHAPE
            )

            # ------------------------------------------------
            # Check label
            # ------------------------------------------------

            expected_label = (
                1
                if label == "error"
                else 0
            )

            label_valid = (
                int(label_value)
                == expected_label
            )

            # ------------------------------------------------
            # Final validation
            # ------------------------------------------------

            if (
                shape_valid
                and label_valid
            ):

                valid += 1

                print(
                    f"{file.name:<50}"
                    f"shape={frames.shape} "
                    f"label={int(label_value)} "
                    f"recording={recording_id} "
                    f"step={step_id}"
                )

            else:

                invalid += 1

                print(
                    f"INVALID: {file.name}"
                )

                print(
                    f"  Shape : {frames.shape}"
                )

                print(
                    f"  Label : {label_value}"
                )

        except Exception as e:

            invalid += 1

            print(
                f"ERROR: {file.name}"
            )

            print(
                f"  {e}"
            )

    print(
        f"\nValid sequences   : "
        f"{valid}"
    )

    print(
        f"Invalid sequences : "
        f"{invalid}"
    )

    return valid, invalid


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI")
    print(" ANNOTATED SEQUENCE INSPECTOR")
    print("==========================================")

    print(
        f"\nExpected shape: "
        f"{EXPECTED_SHAPE}"
    )

    # --------------------------------------------------------
    # Correct
    # --------------------------------------------------------

    correct_valid, correct_invalid = (
        inspect_class("correct")
    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error_valid, error_invalid = (
        inspect_class("error")
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_valid = (
        correct_valid
        + error_valid
    )

    total_invalid = (
        correct_invalid
        + error_invalid
    )

    total = (
        total_valid
        + total_invalid
    )

    print("\n==========================================")
    print("ANNOTATED SEQUENCE DATASET SUMMARY")
    print("==========================================")

    print(
        f"Correct sequences : "
        f"{correct_valid + correct_invalid}"
    )

    print(
        f"Correct valid     : "
        f"{correct_valid}"
    )

    print(
        f"Error sequences   : "
        f"{error_valid + error_invalid}"
    )

    print(
        f"Error valid       : "
        f"{error_valid}"
    )

    print(
        f"Total sequences   : "
        f"{total}"
    )

    print(
        f"Valid sequences   : "
        f"{total_valid}"
    )

    print(
        f"Invalid sequences : "
        f"{total_invalid}"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n==========================================")

    if total_invalid == 0 and total > 0:

        print(
            "ANNOTATED SEQUENCE DATASET CHECK PASSED."
        )

    elif total == 0:

        print(
            "ERROR: No sequences found."
        )

    else:

        print(
            "WARNING: Invalid sequences found."
        )

    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()