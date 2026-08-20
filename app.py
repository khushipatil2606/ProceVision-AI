import sys
import tempfile
import json
import re
from pathlib import Path
from io import BytesIO

import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# MODEL
# ============================================================

from temporal.video_temporal_predict_v3 import analyze_video


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ProceVision AI",
    page_icon="🎥",
    layout="wide"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_recording_id(video_name):
    """
    Example:
        16_2_360p.mp4 -> 16_2
    """
    stem = Path(video_name).stem

    match = re.match(r"^(\d+_\d+)", stem)

    if match:
        return match.group(1)

    return stem.replace("_360p", "")


def load_step_report(recording_id):
    """
    Load the V3 step-level JSON report.
    """

    report_path = (
        PROJECT_ROOT
        / "outputs"
        / "step_predictions"
        / f"{recording_id}_v3_step_report.json"
    )

    if not report_path.exists():
        return None, report_path

    try:

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file), report_path

    except Exception:

        return None, report_path


def extract_step_records(report):
    """
    Find step records regardless of the top-level
    JSON key used by the step predictor.
    """

    if not isinstance(report, dict):
        return []

    possible_keys = (
        "steps",
        "step_analysis",
        "step_predictions",
        "results",
        "step_results"
    )

    for key in possible_keys:

        value = report.get(key)

        if isinstance(value, list):

            return [
                item
                for item in value
                if isinstance(item, dict)
            ]

    # Fallback search
    for value in report.values():

        if not isinstance(value, list):
            continue

        dictionaries = [
            item
            for item in value
            if isinstance(item, dict)
        ]

        if not dictionaries:
            continue

        if any(
            any(
                key in item
                for key in (
                    "step_id",
                    "step_number",
                    "step",
                    "prediction"
                )
            )
            for item in dictionaries
        ):

            return dictionaries

    return []


def normalize_step(step, index):

    step_number = step.get(
        "step_number",
        step.get(
            "step",
            index + 1
        )
    )

    step_id = step.get(
        "step_id",
        step.get(
            "id",
            "N/A"
        )
    )

    prediction = str(
        step.get(
            "prediction",
            step.get(
                "result",
                step.get(
                    "status",
                    "UNKNOWN"
                )
            )
        )
    ).upper()

    start = _as_float(
        step.get(
            "start_time",
            step.get(
                "start",
                step.get(
                    "start_seconds",
                    0
                )
            )
        )
    )

    end = _as_float(
        step.get(
            "end_time",
            step.get(
                "end",
                step.get(
                    "end_seconds",
                    0
                )
            )
        )
    )

    confidence = _as_float(
        step.get(
            "confidence",
            step.get(
                "score",
                step.get(
                    "probability",
                    0
                )
            )
        )
    )

    # Convert 0-100 confidence to 0-1
    if confidence > 1:
        confidence /= 100.0

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    description = str(
        step.get(
            "description",
            step.get(
                "instruction",
                step.get(
                    "text",
                    "No description available."
                )
            )
        )
    )

    return {
        "step_number": step_number,
        "step_id": step_id,
        "prediction": prediction,
        "start": start,
        "end": end,
        "confidence": confidence,
        "description": description,
    }


def find_matching_error_steps(
    report,
    error_regions
):
    """
    Match ERROR steps with temporal error regions.

    IMPORTANT:
    A single procedural step may overlap multiple
    temporal ERROR sequences.

    Therefore we return unique STEP records,
    not one record per temporal sequence.
    """

    records = extract_step_records(report)

    normalized_steps = [
        normalize_step(step, index)
        for index, step in enumerate(records)
    ]

    matched = []

    for region in error_regions:

        region_start = _as_float(
            region.get(
                "start",
                region.get(
                    "start_time",
                    0
                )
            )
        )

        region_end = _as_float(
            region.get(
                "end",
                region.get(
                    "end_time",
                    0
                )
            )
        )

        for step in normalized_steps:

            step_start = step["start"]
            step_end = step["end"]

            # Temporal overlap
            overlaps = (
                step_start <= region_end
                and
                step_end >= region_start
            )

            # Actual procedural ERROR
            is_error = step["prediction"] in {
                "ERROR",
                "ERR",
                "INCORRECT"
            }

            if overlaps and is_error:

                unique_key = (
                    str(step["step_number"]),
                    str(step["step_id"]),
                    round(step_start, 3),
                    round(step_end, 3)
                )

                existing_keys = {
                    (
                        str(item["step_number"]),
                        str(item["step_id"]),
                        round(item["start"], 3),
                        round(item["end"], 3)
                    )
                    for item in matched
                }

                if unique_key not in existing_keys:

                    matched.append(step)

    return matched


def merge_error_regions(regions):
    """
    Merge overlapping or directly adjacent temporal
    error regions into a single continuous region.

    Example:

        992-1006
        1008-1022
        1024-1038
        1040-1054
        1056-1070

    becomes:

        992-1070
    """

    if not regions:
        return []

    cleaned = []

    for region in regions:

        start = _as_float(
            region.get(
                "start",
                region.get(
                    "start_time",
                    0
                )
            )
        )

        end = _as_float(
            region.get(
                "end",
                region.get(
                    "end_time",
                    0
                )
            )
        )

        confidence = _as_float(
            region.get(
                "confidence",
                region.get(
                    "score",
                    0
                )
            )
        )

        if confidence > 1:
            confidence /= 100.0

        cleaned.append({
            "start": start,
            "end": end,
            "confidence": confidence
        })

    cleaned.sort(
        key=lambda x: x["start"]
    )

    merged = []

    for region in cleaned:

        if not merged:

            merged.append(region)
            continue

        previous = merged[-1]

        # Merge overlapping or nearly adjacent regions.
        if region["start"] <= previous["end"] + 2:

            previous["end"] = max(
                previous["end"],
                region["end"]
            )

            previous["confidence"] = max(
                previous["confidence"],
                region["confidence"]
            )

        else:

            merged.append(region)

    return merged


def calculate_confidences(
    predictions,
    scores
):

    confidence_values = []

    correct_confidences = []
    error_confidences = []

    for prediction, score in zip(
        predictions,
        scores
    ):

        try:

            prediction = int(prediction)
            score = float(score)

        except (
            TypeError,
            ValueError
        ):

            continue

        score = max(
            0.0,
            min(1.0, score)
        )

        # Score = probability of ERROR
        if prediction == 1:

            confidence = score
            error_confidences.append(
                confidence
            )

        else:

            confidence = 1.0 - score
            correct_confidences.append(
                confidence
            )

        confidence_values.append(
            confidence
        )

    average_confidence = (
        sum(confidence_values)
        / len(confidence_values)
        if confidence_values
        else 0.0
    )

    avg_correct_confidence = (
        sum(correct_confidences)
        / len(correct_confidences)
        if correct_confidences
        else 0.0
    )

    avg_error_confidence = (
        sum(error_confidences)
        / len(error_confidences)
        if error_confidences
        else 0.0
    )

    return (
        confidence_values,
        average_confidence,
        avg_correct_confidence,
        avg_error_confidence
    )


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 30px;
    }

    .success-box {
        padding: 25px;
        border-radius: 12px;
        background-color: #123d29;
        border: 1px solid #22c55e;
        color: white;
        text-align: center;
        font-size: 25px;
        font-weight: 700;
    }

    .error-box {
        padding: 25px;
        border-radius: 12px;
        background-color: #4a1717;
        border: 1px solid #ef4444;
        color: white;
        text-align: center;
        font-size: 25px;
        font-weight: 700;
    }

    .warning-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #422006;
        border: 1px solid #f59e0b;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ System Information")

    st.markdown(
        """
        **CNN-LSTM V3**

        Temporal Video Analysis

        Step-Level Error Detection
        """
    )

    st.divider()

    st.success(
        "CNN-LSTM V3 Model Available"
    )

    st.info(
        "Upload a procedural video "
        "to analyze it using the "
        "CNN-LSTM temporal model."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎥 ProceVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Procedural Error Detection '
    'using CNN-LSTM'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# VIDEO UPLOAD
# ============================================================

st.header(
    "📹 Video Temporal Analysis"
)

st.write(
    "Upload a procedure video and the "
    "CNN-LSTM V3 model will analyze "
    "the temporal sequence for "
    "procedural errors."
)

uploaded_video = st.file_uploader(
    "Upload Procedure Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_video is not None:

    st.video(uploaded_video)

    st.write(
        f"**Selected video:** "
        f"{uploaded_video.name}"
    )

    if st.button(
        "🚀 Analyze Video",
        type="primary",
        use_container_width=True
    ):

        try:

            suffix = Path(
                uploaded_video.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_video.getbuffer()
                )

                temp_video_path = Path(
                    temp_file.name
                )

            progress = st.progress(0)
            status = st.empty()

            status.info(
                "Loading CNN-LSTM V3 model..."
            )

            progress.progress(20)

            status.info(
                "Extracting video frames..."
            )

            progress.progress(40)

            result = analyze_video(
                temp_video_path
            )

            progress.progress(100)

            status.success(
                "Video analysis completed."
            )

            st.session_state[
                "video_result"
            ] = result

        except Exception as e:

            st.error(
                f"Analysis failed: {e}"
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "video_result" in st.session_state:

    result = st.session_state[
        "video_result"
    ]

    st.divider()

    st.header(
        "📊 Analysis Results"
    )


    # ========================================================
    # RAW MODEL OUTPUT
    # ========================================================

    predictions = result.get(
        "predictions",
        []
    )

    scores = result.get(
        "scores",
        []
    )

    total_frames = int(
        result.get(
            "total_frames",
            0
        )
    )


    # ========================================================
    # NORMALIZE PREDICTIONS
    # ========================================================

    normalized_predictions = []

    for prediction in predictions:

        try:

            normalized_predictions.append(
                int(prediction)
            )

        except (
            TypeError,
            ValueError
        ):

            continue


    # ========================================================
    # SEQUENCE COUNTS
    # ========================================================

    if normalized_predictions:

        total_sequences = len(
            normalized_predictions
        )

        correct_count = sum(
            prediction == 0
            for prediction
            in normalized_predictions
        )

        error_count = sum(
            prediction == 1
            for prediction
            in normalized_predictions
        )

    else:

        total_sequences = int(
            result.get(
                "total_sequences",
                0
            )
        )

        correct_count = int(
            result.get(
                "correct_count",
                0
            )
        )

        error_count = int(
            result.get(
                "error_count",
                0
            )
        )


    correct_percentage = (
        correct_count
        / total_sequences
        * 100
        if total_sequences > 0
        else 0
    )

    error_percentage = (
        error_count
        / total_sequences
        * 100
        if total_sequences > 0
        else 0
    )


    # ========================================================
    # ERROR REGIONS
    # ========================================================

    raw_regions = result.get(
        "error_regions",
        []
    )

    regions = merge_error_regions(
        raw_regions
    )


    # ========================================================
    # STEP LEVEL REPORT
    # ========================================================

    matched_steps = []

    recording_id = None
    step_report_path = None

    if uploaded_video is not None:

        recording_id = get_recording_id(
            uploaded_video.name
        )

        step_report, step_report_path = (
            load_step_report(
                recording_id
            )
        )

        if step_report is not None:

            matched_steps = (
                find_matching_error_steps(
                    step_report,
                    regions
                )
            )


    # ========================================================
    # METRICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Frames",
            total_frames
        )

    with col2:

        st.metric(
            "Temporal Sequences",
            total_sequences
        )

    with col3:

        st.metric(
            "Correct Sequences",
            f"{correct_count} "
            f"({correct_percentage:.1f}%)"
        )

    with col4:

        st.metric(
            "Error Sequences",
            f"{error_count} "
            f"({error_percentage:.1f}%)"
        )


    # ========================================================
    # OVERALL RESULT
    # ========================================================

    st.subheader(
        "🚨 Overall Result"
    )

    if error_count > 0:

        st.markdown(
            """
            <div class="error-box">
                🚨 PROCEDURAL ERROR DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="success-box">
                ✅ NO PROCEDURAL ERROR DETECTED
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    (
        confidence_values,
        average_confidence,
        avg_correct_confidence,
        avg_error_confidence
    ) = calculate_confidences(
        normalized_predictions,
        scores
    )


    st.subheader(
        "🎯 Model Confidence"
    )

    conf1, conf2, conf3 = st.columns(3)

    with conf1:

        st.metric(
            "Average Prediction Confidence",
            f"{average_confidence * 100:.2f}%"
        )

    with conf2:

        st.metric(
            "Correct Confidence",
            f"{avg_correct_confidence * 100:.2f}%"
        )

    with conf3:

        st.metric(
            "Error Confidence",
            f"{avg_error_confidence * 100:.2f}%"
        )


    # ========================================================
    # DETECTED ERROR REGIONS
    # ========================================================

    st.subheader(
        "⏱️ Detected Error Regions"
    )

    if regions:

        for i, region in enumerate(
            regions,
            start=1
        ):

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### 🚨 Error Region {i}"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.write(
                        f"**Start:** "
                        f"{region['start']:.2f}s"
                    )

                with c2:

                    st.write(
                        f"**End:** "
                        f"{region['end']:.2f}s"
                    )

                with c3:

                    st.write(
                        f"**Confidence:** "
                        f"{region['confidence'] * 100:.2f}%"
                    )

    else:

        st.success(
            "✅ No procedural error regions detected."
        )


    # ========================================================
    # STEP LEVEL ERROR DETAILS
    # ========================================================

    st.subheader(
        "📋 Step-Level Error Details"
    )

    if matched_steps:

        st.error(
            f"🚨 {len(matched_steps)} "
            f"actual procedural step-level "
            f"error(s) detected."
        )

        for step in matched_steps:

            with st.container(
                border=True
            ):

                st.markdown(
                    f"### 🚨 Step "
                    f"{step['step_number']}"
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.write(
                        f"**Step ID:** "
                        f"{step['step_id']}"
                    )

                    st.write(
                        "**Status:** 🔴 ERROR"
                    )

                with c2:

                    st.write(
                        f"**Time:** "
                        f"{step['start']:.2f}s - "
                        f"{step['end']:.2f}s"
                    )

                    st.write(
                        f"**Confidence:** "
                        f"{step['confidence'] * 100:.2f}%"
                    )

                with c3:

                    st.write(
                        "**Instruction / Description:**"
                    )

                    st.info(
                        step["description"]
                    )

    elif regions:

        st.warning(
            "⚠️ Temporal error regions were "
            "detected, but no matching ERROR "
            "step was found in the step-level report."
        )

    else:

        st.success(
            "✅ No procedural errors were "
            "detected at the step level."
        )


    # ========================================================
    # TEMPORAL SEQUENCE ANALYSIS
    # ========================================================

    st.subheader(
        "🔍 Temporal Sequence Analysis"
    )

    sequence_data = []

    sequence_times = result.get(
        "sequence_times",
        []
    )


    for i, prediction in enumerate(
        normalized_predictions
    ):

        if i < len(sequence_times):

            start, end = sequence_times[i]

        else:

            start = 0
            end = 0


        try:

            score = (
                float(scores[i])
                if i < len(scores)
                else 0.0
            )

        except (
            TypeError,
            ValueError
        ):

            score = 0.0


        prediction = int(
            prediction
        )


        # Score = probability of ERROR
        if prediction == 1:

            confidence = score
            prediction_label = "ERROR"

        else:

            confidence = 1.0 - score
            prediction_label = "CORRECT"


        confidence = max(
            0.0,
            min(1.0, confidence)
        )


        sequence_data.append(
            {
                "Sequence": i + 1,
                "Start (s)": round(
                    float(start),
                    2
                ),
                "End (s)": round(
                    float(end),
                    2
                ),
                "Prediction": prediction_label,
                "Confidence": (
                    f"{confidence * 100:.2f}%"
                )
            }
        )


    st.dataframe(
        sequence_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # ERROR SEQUENCE NAVIGATOR
    # ========================================================

    if error_count > 0 and uploaded_video is not None:

        st.subheader(
            "🎬 Error Sequence Navigator"
        )

        st.info(
            "Select an error sequence to review "
            "the corresponding section of the video."
        )

        # Use the already-merged temporal error regions instead of
        # showing every individual CNN-LSTM ERROR sequence.
        # This makes one continuous procedural error appear as ONE
        # error region in the navigator.
        error_sequences = regions


        if error_sequences:

            sequence_options = []

            for index, row in enumerate(
                error_sequences,
                start=1
            ):

                start_time = _as_float(
                    row.get("start", 0)
                )

                end_time = _as_float(
                    row.get("end", 0)
                )

                confidence = _as_float(
                    row.get("confidence", 0)
                )

                sequence_options.append(
                    (
                        f"Error Region "
                        f"{index} — "
                        f"{start_time:.2f}s to "
                        f"{end_time:.2f}s — "
                        f"{confidence * 100:.2f}%",
                        row
                    )
                )


            selected_label = st.selectbox(
                "Select an error region",
                [
                    item[0]
                    for item
                    in sequence_options
                ],
                key="error_sequence_navigator"
            )


            selected_row = next(
                item[1]
                for item
                in sequence_options
                if item[0] == selected_label
            )


            selected_region_number = (
                [
                    item[0]
                    for item
                    in sequence_options
                ].index(selected_label)
                + 1
            )

            selected_start = _as_float(
                selected_row.get("start", 0)
            )

            selected_end = _as_float(
                selected_row.get("end", 0)
            )

            selected_confidence = _as_float(
                selected_row.get("confidence", 0)
            )


            st.markdown(
                f"### 🚨 Error Region "
                f"{selected_region_number}"
            )


            nav1, nav2, nav3 = st.columns(3)

            with nav1:

                st.write(
                    f"**Detected:** "
                    f"{selected_start:.2f}s – "
                    f"{selected_end:.2f}s"
                )

            with nav2:

                st.write(
                    f"**Confidence:** "
                    f"{selected_confidence * 100:.2f}%"
                )

            with nav3:

                evidence_start = max(
                    0.0,
                    selected_start - 5.0
                )

                st.write(
                    f"**Evidence starts:** "
                    f"{evidence_start:.2f}s"
                )


            st.subheader(
                "🎥 Video Evidence"
            )

            st.caption(
                "Playback starts approximately "
                "5 seconds before the detected "
                "error region."
            )


            try:

                st.video(
                    uploaded_video,
                    start_time=int(
                        evidence_start
                    )
                )

            except Exception:

                st.video(
                    uploaded_video
                )

                st.warning(
                    "Your Streamlit version does "
                    "not support timestamped playback."
                )


            # ====================================================
            # MATCH STEP
            # ====================================================

            selected_step = None

            for step in matched_steps:

                if (
                    step["start"]
                    <= selected_end
                    and
                    step["end"]
                    >= selected_start
                ):

                    selected_step = step
                    break


            if selected_step:

                st.subheader(
                    "📋 Related Procedural Step"
                )

                with st.container(
                    border=True
                ):

                    c1, c2 = st.columns(2)

                    with c1:

                        st.markdown(
                            f"### 🚨 Step "
                            f"{selected_step['step_number']}"
                        )

                        st.write(
                            f"**Step ID:** "
                            f"{selected_step['step_id']}"
                        )

                        st.write(
                            "**Status:** 🔴 ERROR"
                        )

                    with c2:

                        st.write(
                            f"**Time:** "
                            f"{selected_step['start']:.2f}s – "
                            f"{selected_step['end']:.2f}s"
                        )

                        st.write(
                            f"**Confidence:** "
                            f"{selected_step['confidence'] * 100:.2f}%"
                        )

                    st.write(
                        "**Instruction / Description:**"
                    )

                    st.info(
                        selected_step["description"]
                    )


    # ========================================================
    # ERROR SUMMARY
    # ========================================================

    st.subheader(
        "🚨 Error Summary"
    )

    if error_count > 0:

        st.error(
            f"{error_count} temporal sequence(s) "
            f"were classified as ERROR."
        )

        st.write(
            f"The CNN-LSTM model detected "
            f"{len(regions)} continuous "
            f"error region(s)."
        )

        if matched_steps:

            st.write(
                f"The temporal errors correspond "
                f"to {len(matched_steps)} "
                f"step-level procedural error(s)."
            )

            for i, step in enumerate(
                matched_steps,
                start=1
            ):

                st.write(
                    f"**Procedural Error {i}:** "
                    f"Step {step['step_number']} "
                    f"(Step ID {step['step_id']}) | "
                    f"{step['start']:.2f}s - "
                    f"{step['end']:.2f}s"
                )

        if regions:

            st.write(
                "**Detected temporal region(s):**"
            )

            for i, region in enumerate(
                regions,
                start=1
            ):

                st.write(
                    f"Region {i}: "
                    f"{region['start']:.2f}s - "
                    f"{region['end']:.2f}s | "
                    f"Confidence: "
                    f"{region['confidence'] * 100:.2f}%"
                )

    else:

        st.subheader(
            "✅ Error Summary"
        )

        st.success(
            "The CNN-LSTM V3 model did not detect "
            "any procedural errors in the analyzed video."
        )


    # ========================================================
    # REPORT DATA
    # ========================================================

    report_data = {

        "application":
            "ProceVision AI",

        "model":
            "CNN-LSTM V3",

        "analysis_type":
            "Temporal Procedural Error Detection",

        "video":
            uploaded_video.name
            if uploaded_video is not None
            else "Analyzed Video",

        "total_frames":
            total_frames,

        "total_sequences":
            total_sequences,

        "correct_sequences":
            correct_count,

        "error_sequences":
            error_count,

        "correct_percentage":
            round(
                correct_percentage,
                2
            ),

        "error_percentage":
            round(
                error_percentage,
                2
            ),

        "average_confidence":
            round(
                average_confidence * 100,
                2
            ),

        "correct_confidence":
            round(
                avg_correct_confidence * 100,
                2
            ),

        "error_confidence":
            round(
                avg_error_confidence * 100,
                2
            ),

        "detected_error_regions":
            len(regions),

        "step_level_errors":
            len(matched_steps),

        "overall_result":
            (
                "PROCEDURAL ERROR DETECTED"
                if error_count > 0
                else
                "NO PROCEDURAL ERROR DETECTED"
            ),

        "error_regions":
            regions,

        "matched_step_errors":
            matched_steps,

        "sequence_analysis":
            sequence_data
    }


    # ========================================================
    # JSON DOWNLOAD
    # ========================================================

    st.subheader(
        "📥 Download Analysis Report"
    )

    report_json = json.dumps(
        report_data,
        indent=4,
        default=str
    )

    st.download_button(
        label="⬇️ Download JSON Report",
        data=report_json,
        file_name=(
            "procevision_analysis_report.json"
        ),
        mime="application/json",
        key="download_json_report"
    )


    # ========================================================
    # PDF REPORT
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📄 Professional PDF Report"
    )

    try:

        pdf_buffer = BytesIO()

        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "ProceVisionTitle",
            parent=styles["Title"],
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=8
        )

        subtitle_style = ParagraphStyle(
            "ProceVisionSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#555555"
            ),
            spaceAfter=18
        )

        heading_style = ParagraphStyle(
            "ReportHeading",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8
        )

        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            spaceAfter=5
        )

        story = []


        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "ProceVision AI",
                title_style
            )
        )

        story.append(
            Paragraph(
                "CNN-LSTM V3 | "
                "Temporal Procedural Error Detection",
                subtitle_style
            )
        )


        # ----------------------------------------------------
        # VIDEO INFO
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "Video Information",
                heading_style
            )
        )

        video_name = (
            uploaded_video.name
            if uploaded_video is not None
            else "Analyzed Video"
        )

        video_table = Table(
            [
                ["Field", "Value"],
                [
                    "Video",
                    str(video_name)
                ],
                [
                    "Model",
                    "CNN-LSTM V3"
                ],
                [
                    "Analysis",
                    "Temporal Procedural Error Detection"
                ]
            ],
            colWidths=[
                1.7 * inch,
                4.8 * inch
            ]
        )

        video_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f2937")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ])
        )

        story.append(
            video_table
        )


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "Analysis Summary",
                heading_style
            )
        )

        summary_table = Table(
            [
                ["Metric", "Value"],

                [
                    "Total Frames",
                    str(total_frames)
                ],

                [
                    "Temporal Sequences",
                    str(total_sequences)
                ],

                [
                    "Correct Sequences",
                    f"{correct_count} "
                    f"({correct_percentage:.1f}%)"
                ],

                [
                    "Error Sequences",
                    f"{error_count} "
                    f"({error_percentage:.1f}%)"
                ],

                [
                    "Detected Error Regions",
                    str(len(regions))
                ],

                [
                    "Step-Level Procedural Errors",
                    str(len(matched_steps))
                ],

                [
                    "Overall Result",
                    (
                        "PROCEDURAL ERROR DETECTED"
                        if error_count > 0
                        else
                        "NO PROCEDURAL ERROR DETECTED"
                    )
                ]
            ],
            colWidths=[
                3.1 * inch,
                3.4 * inch
            ]
        )

        summary_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f2937")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ])
        )

        story.append(
            summary_table
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "Model Confidence",
                heading_style
            )
        )

        confidence_table = Table(
            [
                [
                    "Confidence Metric",
                    "Value"
                ],

                [
                    "Average Prediction Confidence",
                    f"{average_confidence * 100:.2f}%"
                ],

                [
                    "Correct Prediction Confidence",
                    f"{avg_correct_confidence * 100:.2f}%"
                ],

                [
                    "Error Prediction Confidence",
                    f"{avg_error_confidence * 100:.2f}%"
                ]
            ],
            colWidths=[
                3.1 * inch,
                3.4 * inch
            ]
        )

        confidence_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#1f2937")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                )
            ])
        )

        story.append(
            confidence_table
        )


        # ----------------------------------------------------
        # ERROR REGIONS
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "Detected Error Regions",
                heading_style
            )
        )

        if regions:

            region_data = [
                [
                    "Region",
                    "Start (s)",
                    "End (s)",
                    "Confidence"
                ]
            ]

            for i, region in enumerate(
                regions,
                start=1
            ):

                region_data.append(
                    [
                        f"Error Region {i}",
                        f"{region['start']:.2f}",
                        f"{region['end']:.2f}",
                        f"{region['confidence'] * 100:.2f}%"
                    ]
                )

            region_table = Table(
                region_data,
                repeatRows=1,
                colWidths=[
                    1.8 * inch,
                    1.5 * inch,
                    1.5 * inch,
                    1.7 * inch
                ]
            )

            region_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#7f1d1d")
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8
                    ),
                    (
                        "ALIGN",
                        (1, 1),
                        (-1, -1),
                        "CENTER"
                    )
                ])
            )

            story.append(
                region_table
            )

        else:

            story.append(
                Paragraph(
                    "No procedural error regions were detected.",
                    body_style
                )
            )


        # ----------------------------------------------------
        # STEP LEVEL ERRORS
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "Step-Level Procedural Errors",
                heading_style
            )
        )

        if matched_steps:

            step_data = [
                [
                    "Step",
                    "Step ID",
                    "Start",
                    "End",
                    "Status",
                    "Confidence"
                ]
            ]

            for step in matched_steps:

                step_data.append(
                    [
                        str(
                            step["step_number"]
                        ),
                        str(
                            step["step_id"]
                        ),
                        f"{step['start']:.2f}s",
                        f"{step['end']:.2f}s",
                        "ERROR",
                        f"{step['confidence'] * 100:.2f}%"
                    ]
                )

            step_table = Table(
                step_data,
                repeatRows=1,
                colWidths=[
                    0.65 * inch,
                    0.75 * inch,
                    1.0 * inch,
                    1.0 * inch,
                    1.05 * inch,
                    1.15 * inch
                ]
            )

            step_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#7f1d1d")
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold"
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7.5
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    )
                ])
            )

            story.append(
                step_table
            )

            story.append(
                Spacer(1, 8)
            )

            for step in matched_steps:

                story.append(
                    Paragraph(
                        f"<b>Step "
                        f"{step['step_number']} "
                        f"Description:</b> "
                        f"{step['description']}",
                        body_style
                    )
                )

        else:

            story.append(
                Paragraph(
                    "No step-level procedural errors "
                    "were matched with the detected "
                    "video regions.",
                    body_style
                )
            )


        # ----------------------------------------------------
        # FINAL INTERPRETATION
        # ----------------------------------------------------

        story.append(
            Paragraph(
                "Final Interpretation",
                heading_style
            )
        )

        if error_count > 0:

            final_text = (
                f"The CNN-LSTM V3 model classified "
                f"{error_count} temporal sequence(s) "
                f"as ERROR. These sequences form "
                f"{len(regions)} continuous error "
                f"region(s) and correspond to "
                f"{len(matched_steps)} step-level "
                f"procedural error(s)."
            )

        else:

            final_text = (
                "The CNN-LSTM V3 model did not "
                "detect any procedural errors "
                "in the analyzed video."
            )

        story.append(
            Paragraph(
                final_text,
                body_style
            )
        )

        story.append(
            Spacer(1, 20)
        )

        story.append(
            Paragraph(
                "Generated by ProceVision AI | CNN-LSTM V3",
                ParagraphStyle(
                    "ReportFooter",
                    parent=styles["Normal"],
                    fontSize=8,
                    alignment=TA_CENTER,
                    textColor=colors.HexColor(
                        "#666666"
                    )
                )
            )
        )


        doc.build(story)

        pdf_buffer.seek(0)

        st.download_button(
            label="📄 Download Professional PDF Report",
            data=pdf_buffer.getvalue(),
            file_name=(
                "ProceVision_AI_Analysis_Report.pdf"
            ),
            mime="application/pdf",
            key="download_pdf_report"
        )

    except Exception as pdf_error:

        st.error(
            f"Unable to generate PDF report: "
            f"{pdf_error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ProceVision AI | "
    "CNN-LSTM V3 | "
    "Temporal Procedural Error Detection"
)