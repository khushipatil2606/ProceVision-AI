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

TARGET_RECORDING = "16_2"


# ============================================
# MAIN
# ============================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - ERROR INSPECTOR")
    print("==========================================")

    print(f"\nTarget recording: {TARGET_RECORDING}")

    if not ANNOTATION_FILE.exists():

        print("\nERROR: Annotation file not found.")
        return

    with open(
        ANNOTATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    result = None

    for item in data:

        if str(item.get("recording_id")) == TARGET_RECORDING:

            result = item
            break

    if result is None:

        print("\nERROR: Recording not found.")
        return

    print("\n==========================================")
    print("RECORDING INFORMATION")
    print("==========================================")

    print(f"Recording ID : {result.get('recording_id')}")
    print(f"Activity ID  : {result.get('activity_id')}")
    print(f"Is Error     : {result.get('is_error')}")

    step_annotations = result.get(
        "step_annotations",
        []
    )

    print("\n==========================================")
    print("STEP ANNOTATIONS")
    print("==========================================")

    for index, step in enumerate(step_annotations):

        print(f"\nSTEP {index + 1}")

        print(
            f"Description : "
            f"{step.get('description')}"
        )

        print(
            f"Step ID     : "
            f"{step.get('step_id')}"
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
    print("RAW ERROR INFORMATION")
    print("==========================================")

    print(
        json.dumps(
            result,
            indent=2
        )
    )

    print("\n==========================================")
    print("Inspection completed.")
    print("==========================================\n")


if __name__ == "__main__":
    main()