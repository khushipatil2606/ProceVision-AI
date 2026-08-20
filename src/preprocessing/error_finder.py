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
    / "error_annotations.json"
)

# Scrambled Eggs
TARGET_ACTIVITY_ID = 16


# ============================================
# MAIN
# ============================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - SCRAMBLED EGGS")
    print("        ERROR RECORDING FINDER")
    print("==========================================")

    if not ANNOTATION_FILE.exists():

        print("\nERROR: error_annotations.json not found.")
        return

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # Find error recordings for Activity 16
    error_videos = []

    for item in data:

        if (
            item.get("activity_id") == TARGET_ACTIVITY_ID
            and item.get("is_error") is True
        ):

            error_videos.append(item)

    print(
        f"\nScrambled Eggs error recordings found: "
        f"{len(error_videos)}"
    )

    print("\n==========================================")
    print("RECORDINGS")
    print("==========================================")

    for index, item in enumerate(error_videos):

        print(
            f"{index + 1:02d}. "
            f"Recording: {item.get('recording_id')} | "
            f"Activity ID: {item.get('activity_id')}"
        )

    print("\n==========================================")
    print("REFERENCE CORRECT RECORDING")
    print("==========================================")
    print("Recording: 16_1")
    print("Activity: Scrambled Eggs")
    print("Status: CORRECT")
    print("==========================================\n")


if __name__ == "__main__":
    main()