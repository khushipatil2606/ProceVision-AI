import json
from pathlib import Path


# ============================================
# PROJECT PATH
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "annotation_json"
    / "complete_step_annotations.json"
)

RECORDINGS = ["16_1", "16_2"]

# Step 20 = step_id 170
TARGET_STEP_ID = 170


# ============================================
# LOAD JSON
# ============================================

def load_json():

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================
# FIND RECORDING RECURSIVELY
# ============================================

def find_recording(data, recording_id):

    if isinstance(data, dict):

        # Check whether this dictionary represents
        # the recording we want.
        if str(data.get("recording_id")) == recording_id:
            return data

        # Search inside nested dictionaries/lists.
        for value in data.values():

            result = find_recording(
                value,
                recording_id
            )

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = find_recording(
                item,
                recording_id
            )

            if result is not None:
                return result

    return None


# ============================================
# FIND STEP
# ============================================

def find_step(recording, step_id):

    if not recording:
        return None

    steps = recording.get("steps", [])

    for step in steps:

        if step.get("step_id") == step_id:
            return step

    return None


# ============================================
# MAIN
# ============================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - STEP COMPARATOR")
    print("==========================================")

    if not ANNOTATION_FILE.exists():

        print("\nERROR: Annotation file not found.")
        print(ANNOTATION_FILE)
        return

    data = load_json()

    print(f"\nTarget Step ID: {TARGET_STEP_ID}")
    print("Description: Cook covered for 1 minute")
    print("\n------------------------------------------")

    for recording_id in RECORDINGS:

        print("\n==========================================")
        print(f"Recording: {recording_id}")
        print("==========================================")

        recording = find_recording(
            data,
            recording_id
        )

        if recording is None:

            print("Recording not found.")
            continue

        step = find_step(
            recording,
            TARGET_STEP_ID
        )

        if step is None:

            print(
                f"Step ID {TARGET_STEP_ID} "
                "not found."
            )

            continue

        print(
            f"Step ID     : "
            f"{step.get('step_id')}"
        )

        print(
            f"Description : "
            f"{step.get('description')}"
        )

        print(
            f"Start time  : "
            f"{step.get('start_time')} seconds"
        )

        print(
            f"End time    : "
            f"{step.get('end_time')} seconds"
        )

    print("\n==========================================")
    print("Comparison completed.")
    print("==========================================\n")


if __name__ == "__main__":
    main()