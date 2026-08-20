import json
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# PATHS
# ============================================================

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "annotation_json"
    / "complete_step_annotations.json"
)

ERROR_FILE = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "annotation_json"
    / "error_annotations.json"
)

VIDEO_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scrambled_eggs"
)


# ============================================================
# SETTINGS
# ============================================================

MIN_STEP_DURATION = 10


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# FIND RECORDINGS
# ============================================================

def find_recordings(data, results):

    """
    Recursively find recording objects.

    Expected structure:

    {
        "recording_id": "16_1",
        "activity_id": 16,
        "step_annotations": [...]
    }
    """

    if isinstance(data, dict):

        recording_id = data.get("recording_id")

        if recording_id is not None:

            results[str(recording_id)] = data

        for value in data.values():

            find_recordings(
                value,
                results
            )

    elif isinstance(data, list):

        for item in data:

            find_recordings(
                item,
                results
            )


# ============================================================
# GET STEP ANNOTATIONS
# ============================================================

def get_step_annotations(record):

    """
    Handles both possible structures:

    step_annotations
    OR
    steps
    """

    steps = record.get(
        "step_annotations"
    )

    if steps is None:

        steps = record.get(
            "steps",
            []
        )

    if not isinstance(steps, list):

        return []

    return steps


# ============================================================
# VALIDATE STEPS
# ============================================================

def get_valid_steps(record):

    steps = get_step_annotations(
        record
    )

    valid_steps = []

    for step in steps:

        if not isinstance(step, dict):

            continue

        start = step.get(
            "start_time",
            -1
        )

        end = step.get(
            "end_time",
            -1
        )

        # ----------------------------------------------------
        # Validate timestamps
        # ----------------------------------------------------

        if not isinstance(
            start,
            (int, float)
        ):

            continue

        if not isinstance(
            end,
            (int, float)
        ):

            continue

        if start < 0:

            continue

        if end <= start:

            continue

        duration = end - start

        if duration < MIN_STEP_DURATION:

            continue

        # ----------------------------------------------------
        # Save only useful information
        # ----------------------------------------------------

        valid_steps.append({

            "step_id": step.get(
                "step_id"
            ),

            "description": step.get(
                "description",
                ""
            ),

            "start_time": float(
                start
            ),

            "end_time": float(
                end
            ),

            "duration": float(
                duration
            )
        })

    return valid_steps


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==========================================")
    print(" PROCEVISION AI - TEMPORAL DATA FINDER")
    print("==========================================")

    # ========================================================
    # LOAD COMPLETE STEP ANNOTATIONS
    # ========================================================

    print("\nLoading step annotations...")

    annotation_data = load_json(
        ANNOTATION_FILE
    )

    recordings = {}

    find_recordings(
        annotation_data,
        recordings
    )

    print(
        f"Annotated recordings found: "
        f"{len(recordings)}"
    )

    # ========================================================
    # LOAD ERROR ANNOTATIONS
    # ========================================================

    print("\nLoading error annotations...")

    error_data = load_json(
        ERROR_FILE
    )

    error_recordings = {}

    find_recordings(
        error_data,
        error_recordings
    )

    print(
        f"Error metadata recordings found: "
        f"{len(error_recordings)}"
    )

    # ========================================================
    # BUILD ERROR STATUS
    # ========================================================

    error_status = {}

    for recording_id, record in error_recordings.items():

        error_status[recording_id] = (
            record.get(
                "is_error",
                False
            ) is True
        )

    # ========================================================
    # ANALYZE RECORDINGS
    # ========================================================

    usable = []

    for recording_id, record in recordings.items():

        # ----------------------------------------------------
        # VIDEO PATH
        # ----------------------------------------------------

        video_path = (
            VIDEO_DIR
            / f"{recording_id}_360p.mp4"
        )

        if not video_path.exists():

            continue

        # ----------------------------------------------------
        # VALID STEPS
        # ----------------------------------------------------

        valid_steps = get_valid_steps(
            record
        )

        if not valid_steps:

            continue

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        is_error = error_status.get(
            recording_id,
            False
        )

        if is_error:

            label = "error"

        else:

            label = "correct"

        # ----------------------------------------------------
        # CREATE RECORD
        # ----------------------------------------------------

        usable.append({

            "recording_id": recording_id,

            "activity_id": record.get(
                "activity_id"
            ),

            "label": label,

            "is_error": is_error,

            "video": str(
                video_path
            ),

            "valid_steps": len(
                valid_steps
            ),

            "steps": valid_steps
        })

    # ========================================================
    # SUMMARY
    # ========================================================

    correct = [
        r
        for r in usable
        if r["label"] == "correct"
    ]

    error = [
        r
        for r in usable
        if r["label"] == "error"
    ]

    print("\n==========================================")
    print("TEMPORAL DATASET SUMMARY")
    print("==========================================")

    print(
        f"Usable recordings : "
        f"{len(usable)}"
    )

    print(
        f"Correct recordings: "
        f"{len(correct)}"
    )

    print(
        f"Error recordings  : "
        f"{len(error)}"
    )

    # ========================================================
    # CORRECT RECORDINGS
    # ========================================================

    print("\n==========================================")
    print("CORRECT RECORDINGS")
    print("==========================================")

    for item in correct:

        print(
            f"{item['recording_id']:<10}"
            f"steps={item['valid_steps']}"
        )

    # ========================================================
    # ERROR RECORDINGS
    # ========================================================

    print("\n==========================================")
    print("ERROR RECORDINGS")
    print("==========================================")

    for item in error:

        print(
            f"{item['recording_id']:<10}"
            f"steps={item['valid_steps']}"
        )

    # ========================================================
    # SPECIFIC RECORDING CHECK
    # ========================================================

    print("\n==========================================")
    print("TEMPORAL RECORDING CHECK")
    print("==========================================")

    for recording_id in ["16_1", "16_2"]:

        matches = [
            r
            for r in usable
            if r["recording_id"] == recording_id
        ]

        if not matches:

            print(
                f"{recording_id}: NOT FOUND"
            )

            continue

        item = matches[0]

        print(
            f"{recording_id}: "
            f"label={item['label']}, "
            f"is_error={item['is_error']}, "
            f"steps={item['valid_steps']}"
        )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "temporal_metadata"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "recordings.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            usable,
            file,
            indent=4
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n==========================================")
    print("METADATA SAVED")
    print("==========================================")

    print(output_file)

    print("\n==========================================")
    print("TEMPORAL DATA FINDER COMPLETE")
    print("==========================================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()