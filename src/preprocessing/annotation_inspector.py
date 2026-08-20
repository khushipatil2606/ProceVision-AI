import json
from pathlib import Path


# ============================================
# PROJECT PATHS
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "annotation_json"
)

RECORDING_ID = "16_1"


# ============================================
# LOAD JSON
# ============================================

def load_json(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================
# FIND RECORDING
# ============================================

def find_recording(data, recording_id):

    results = []

    if isinstance(data, list):

        for item in data:

            if isinstance(item, dict):

                if str(item.get("recording_id")) == recording_id:
                    results.append(item)

    elif isinstance(data, dict):

        # Case where recording IDs are dictionary keys
        if recording_id in data:
            results.append(data[recording_id])

        # Otherwise search recursively
        for value in data.values():

            if isinstance(value, (list, dict)):

                results.extend(
                    find_recording(value, recording_id)
                )

    return results


# ============================================
# PRINT OBJECT
# ============================================

def print_object(obj, level=0):

    indent = "  " * level

    if isinstance(obj, dict):

        for key, value in obj.items():

            print(f"{indent}{key}: ", end="")

            if isinstance(value, (dict, list)):

                print()
                print_object(value, level + 1)

            else:

                print(value)

    elif isinstance(obj, list):

        print(f"{indent}LIST ({len(obj)} items)")

        for index, item in enumerate(obj[:10]):

            print(f"{indent}[{index}]")

            print_object(item, level + 1)

        if len(obj) > 10:

            print(
                f"{indent}... "
                f"{len(obj) - 10} more items"
            )

    else:

        print(f"{indent}{obj}")


# ============================================
# MAIN
# ============================================

def main():

    print("\n==========================================")
    print("   PROCEVISION AI - RECORDING INSPECTOR")
    print("==========================================")

    print(f"\nTarget recording: {RECORDING_ID}")

    files = [
        "complete_step_annotations.json",
        "step_annotations.json",
        "error_annotations.json",
    ]

    for filename in files:

        file_path = ANNOTATION_DIR / filename

        print("\n==========================================")
        print(f"FILE: {filename}")
        print("==========================================")

        if not file_path.exists():

            print("File not found.")
            continue

        try:

            data = load_json(file_path)

            results = find_recording(
                data,
                RECORDING_ID
            )

            print(
                f"\nMatching records found: "
                f"{len(results)}"
            )

            for index, result in enumerate(results):

                print(
                    f"\n---------- RECORD {index + 1} ----------"
                )

                print_object(result)

        except Exception as e:

            print(f"ERROR: {e}")

    print("\n==========================================")
    print("Inspection completed.")
    print("==========================================\n")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    main()