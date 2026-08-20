import sys
import tempfile
from pathlib import Path

import streamlit as st
import json
import re


# ============================================================
# STEP-LEVEL REPORT HELPERS
# ============================================================

def get_recording_id(video_name):
    """Extract recording ID such as 16_2 from 16_2_360p.mp4."""
    stem = Path(video_name).stem
    match = re.match(r"^(\\d+_\\d+)", stem)
    return match.group(1) if match else stem.replace("_360p", "")


def load_step_report(recording_id):
    """Load the step-level V3 JSON report for a recording."""
    report_path = (
        PROJECT_ROOT
        / "outputs"
        / "step_predictions"
        / f"{recording_id}_v3_step_report.json"
    )

    if not report_path.exists():
        return None, report_path

    try:
        with open(report_path, "r", encoding="utf-8") as file:
            return json.load(file), report_path
    except Exception:
        return None, report_path


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_step_records(report):
    """Find the step records regardless of the report's top-level key name."""
    if not isinstance(report, dict):
        return []

    candidates = []
    for key in ("steps", "step_analysis", "step_predictions", "results", "step_results"):
        value = report.get(key)
        if isinstance(value, list):
            candidates = value
            break

    if not candidates:
        # Fallback: find the first list containing step-like dictionaries.
        for value in report.values():
            if isinstance(value, list) and any(isinstance(x, dict) for x in value):
                if any(
                    any(k in x for k in ("step_id", "step_number", "step", "prediction"))
                    for x in value if isinstance(x, dict)
                ):
                    candidates = value
                    break

    return [x for x in candidates if isinstance(x, dict)]


def normalize_step(step, index):
    """Normalize common field-name variants from the step report."""
    step_number = step.get("step_number", step.get("step", index + 1))
    step_id = step.get("step_id", step.get("id", "N/A"))
    prediction = str(
        step.get("prediction", step.get("result", step.get("status", "UNKNOWN")))
    ).upper()

    start = _as_float(
        step.get("start_time", step.get("start", step.get("start_seconds", 0)))
    )
    end = _as_float(
        step.get("end_time", step.get("end", step.get("end_seconds", 0)))
    )
    confidence = _as_float(
        step.get("confidence", step.get("score", step.get("probability", 0)))
    )

    # Reports may store confidence as 0-1 or 0-100.
    if confidence > 1:
        confidence = confidence / 100.0
    confidence = max(0.0, min(1.0, confidence))

    description = str(
        step.get(
            "description",
            step.get("instruction", step.get("text", "No description available.")),
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


def find_matching_error_steps(report, error_regions):
    """Match video-level error regions with overlapping step-level records."""
    records = extract_step_records(report)
    normalized = [normalize_step(step, i) for i, step in enumerate(records)]

    matches = []
    for region in error_regions:
        region_start = _as_float(region.get("start", region.get("start_time", 0)))
        region_end = _as_float(region.get("end", region.get("end_time", 0)))

        for step in normalized:
            # Match only steps predicted as ERROR, and only overlapping time ranges.
            overlaps = step["start"] <= region_end and step["end"] >= region_start
            is_error = step["prediction"] in {"ERROR", "ERR", "INCORRECT"}
            if overlaps and is_error:
                key = (str(step["step_number"]), str(step["step_id"]), step["start"], step["end"])
                if key not in [
                    (str(x["step_number"]), str(x["step_id"]), x["start"], x["end"])
                    for x in matches
                ]:
                    matches.append(step)

    return matches


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ============================================================
# IMPORT MODEL PIPELINE
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
# CUSTOM CSS
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

    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #172554;
        border: 1px solid #3b82f6;
        color: white;
    }

    .confidence-high {
        color: #22c55e;
        font-weight: 700;
    }

    .confidence-error {
        color: #ef4444;
        font-weight: 700;
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
# VIDEO ANALYSIS HEADER
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


# ============================================================
# VIDEO UPLOAD
# ============================================================

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

            # ------------------------------------------------
            # SAVE TEMPORARY VIDEO
            # ------------------------------------------------

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

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            progress = st.progress(0)

            status = st.empty()

            status.info(
                "Loading CNN-LSTM V3 model..."
            )

            progress.progress(20)

            # ------------------------------------------------
            # ANALYSIS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # STORE RESULT
            # ------------------------------------------------

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
    # BASIC METRICS
    # ========================================================

    total_sequences = result.get(
        "total_sequences",
        0
    )

    correct_count = result.get(
        "correct_count",
        0
    )

    error_count = result.get(
        "error_count",
        0
    )

    total_frames = result.get(
        "total_frames",
        0
    )


    correct_percentage = (
        correct_count / total_sequences * 100
        if total_sequences > 0
        else 0
    )

    error_percentage = (
        error_count / total_sequences * 100
        if total_sequences > 0
        else 0
    )


    # ========================================================
    # METRIC CARDS
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
            f"{correct_count} ({correct_percentage:.1f}%)"
        )

    with col4:

        st.metric(
            "Error Sequences",
            f"{error_count} ({error_percentage:.1f}%)"
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
    # CONFIDENCE SUMMARY
    # ========================================================

    predictions = result.get(
        "predictions",
        []
    )

    scores = result.get(
        "scores",
        []
    )


    # --------------------------------------------------------
    # Calculate confidence correctly
    #
    # Model score = probability of ERROR
    #
    # ERROR prediction:
    #     confidence = score
    #
    # CORRECT prediction:
    #     confidence = 1 - score
    # --------------------------------------------------------

    confidence_values = []

    for prediction, score in zip(
        predictions,
        scores
    ):

        score = float(score)

        if int(prediction) == 1:

            confidence = score

        else:

            confidence = 1.0 - score

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        confidence_values.append(
            confidence
        )


    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        average_confidence = 0.0


    # ========================================================
    # CONFIDENCE METRICS
    # ========================================================

    st.subheader(
        "🎯 Model Confidence"
    )

    confidence_col1, confidence_col2, confidence_col3 = st.columns(3)

    with confidence_col1:

        st.metric(
            "Average Prediction Confidence",
            f"{average_confidence * 100:.2f}%"
        )

    with confidence_col2:

        if correct_count > 0:

            correct_confidences = [
                confidence_values[i]
                for i in range(len(predictions))
                if int(predictions[i]) == 0
            ]

            avg_correct_confidence = (
                sum(correct_confidences)
                / len(correct_confidences)
                if correct_confidences
                else 0
            )

        else:

            avg_correct_confidence = 0

        st.metric(
            "Correct Confidence",
            f"{avg_correct_confidence * 100:.2f}%"
        )

    with confidence_col3:

        if error_count > 0:

            error_confidences = [
                confidence_values[i]
                for i in range(len(predictions))
                if int(predictions[i]) == 1
            ]

            avg_error_confidence = (
                sum(error_confidences)
                / len(error_confidences)
                if error_confidences
                else 0
            )

        else:

            avg_error_confidence = 0

        st.metric(
            "Error Confidence",
            f"{avg_error_confidence * 100:.2f}%"
        )


    # ========================================================
    # ERROR REGIONS
    # ========================================================

    st.subheader(
        "⏱️ Detected Error Regions"
    )

    regions = result.get(
        "error_regions",
        []
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

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.write(
                        f"**Start:** "
                        f"{region['start']:.2f}s"
                    )

                with col2:

                    st.write(
                        f"**End:** "
                        f"{region['end']:.2f}s"
                    )

                with col3:

                    st.write(
                        f"**Confidence:** "
                        f"{region['confidence'] * 100:.2f}%"
                    )

    else:

        st.success(
            "✅ No procedural error regions detected."
        )


    # ========================================================\n    # STEP-LEVEL ERROR DETAILS\n    # ========================================================\n\n    st.subheader(\n        "📋 Step-Level Error Details"\n    )\n\n    if uploaded_video is not None:\n\n        recording_id = get_recording_id(\n            uploaded_video.name\n        )\n\n        step_report, step_report_path = load_step_report(\n            recording_id\n        )\n\n        if step_report is not None:\n\n            matched_steps = find_matching_error_steps(\n                step_report,\n                regions\n            )\n\n            if matched_steps:\n\n                st.error(\n                    f"🚨 {len(matched_steps)} "\n                    f"step-level procedural error(s) "\n                    f"matched with the detected video region(s)."\n                )\n\n                for step in matched_steps:\n\n                    with st.container(border=True):\n\n                        st.markdown(\n                            f"### 🚨 Step {step['step_number']}"\n                        )\n\n                        info_col1, info_col2, info_col3 = st.columns(3)\n\n                        with info_col1:\n                            st.write(\n                                f"**Step ID:** {step['step_id']}"\n                            )\n                            st.write(\n                                f"**Status:** 🔴 ERROR"\n                            )\n\n                        with info_col2:\n                            st.write(\n                                f"**Time:** {step['start']:.2f}s - "\n                                f"{step['end']:.2f}s"\n                            )\n                            st.write(\n                                f"**Confidence:** "\n                                f"{step['confidence'] * 100:.2f}%\"\n                            )\n\n                        with info_col3:\n                            st.write(\n                                "**Instruction / Description:**"\n                            )\n                            st.write(\n                                step['description']\n                            )\n\n            elif regions:\n\n                st.warning(\n                    "⚠️ A temporal error region was detected, "\n                    "but no ERROR step in the step-level report "\n                    "overlapped with that region."\n                )\n\n            else:\n\n                st.success(\n                    "✅ No procedural errors were detected "\n                    "at the step level."\n                )\n\n        else:\n\n            st.info(\n                f"ℹ️ Step-level report was not found for "\n                f"**{recording_id}**.\\n\\n"\n                f"Expected file: `{step_report_path}`"\n            )\n\n\n    # ========================================================
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


    for i in range(
        len(predictions)
    ):

        # ----------------------------------------------------
        # Get time range
        # ----------------------------------------------------

        if i < len(sequence_times):

            start, end = sequence_times[i]

        else:

            start = 0
            end = 0


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            predictions[i]
        )


        # ----------------------------------------------------
        # Raw model score
        #
        # score = probability of ERROR
        # ----------------------------------------------------

        score = float(
            scores[i]
        )


        # ----------------------------------------------------
        # CORRECT CONFIDENCE FIX
        # ----------------------------------------------------

        if prediction == 1:

            confidence = score

            prediction_label = "ERROR"

        else:

            confidence = 1.0 - score

            prediction_label = "CORRECT"


        # Make sure confidence stays between 0 and 1

        confidence = max(
            0.0,
            min(1.0, confidence)
        )


        # ----------------------------------------------------
        # Add row
        # ----------------------------------------------------

        sequence_data.append(
            {
                "Sequence":
                    i + 1,

                "Start (s)":
                    round(
                        float(start),
                        2
                    ),

                "End (s)":
                    round(
                        float(end),
                        2
                    ),

                "Prediction":
                    prediction_label,

                "Confidence":
                    f"{confidence * 100:.2f}%"
            }
        )


    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    st.dataframe(
        sequence_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # ERROR SUMMARY
    # ========================================================

    if error_count > 0:

        st.subheader(
            "🚨 Error Summary"
        )

        st.error(
            f"{error_count} temporal sequence(s) "
            f"were classified as procedural errors."
        )

        st.write(
            "The model identified the following "
            "temporal regions as potential errors:"
        )

        for i, region in enumerate(
            regions,
            start=1
        ):

            st.write(
                f"**Error {i}:** "
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
    # DOWNLOAD REPORT
    # ========================================================

    st.subheader(
        "📥 Download Analysis Report"
    )

    report_data = {
        "video": uploaded_video.name
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

        "overall_result":
            (
                "PROCEDURAL ERROR DETECTED"
                if error_count > 0
                else
                "NO PROCEDURAL ERROR DETECTED"
            ),

        "error_regions":
            regions,

        "step_level_errors":
            matched_steps if "matched_steps" in locals() else [],

        "sequence_analysis":
            sequence_data
    }


    report_json = json.dumps(
        report_data,
        indent=4,
        default=str
    )


    st.download_button(
        label="⬇️ Download JSON Report",
        data=report_json,
        file_name="procevision_analysis_report.json",
        mime="application/json"
    )


    # ========================================================
    # FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "ProceVision AI | "
        "CNN-LSTM V3 | "
        "Temporal Procedural Error Detection"
    )