from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)
from reportlab.lib.units import inch
from datetime import datetime


def generate_pdf_report(
    output_path,
    video_name,
    total_frames,
    total_sequences,
    correct_sequences,
    error_sequences,
    error_events,
    avg_confidence,
    error_confidence,
    detected_steps=None
):
    """
    Generate a professional PDF analysis report
    for ProceVision AI.
    """

    if detected_steps is None:
        detected_steps = []

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=28,
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=10,
        textColor=colors.grey,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontSize=15,
        leading=20,
        spaceBefore=15,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        "NormalCustom",
        parent=styles["Normal"],
        fontSize=10,
        leading=15
    )

    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "ProceVision AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Temporal Procedural Error Detection Report",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Video:</b> {video_name}",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            normal_style
        )
    )

    story.append(Spacer(1, 20))

    # --------------------------------------------------
    # OVERALL RESULT
    # --------------------------------------------------

    story.append(
        Paragraph(
            "1. Overall Analysis Result",
            heading_style
        )
    )

    if error_events > 0:
        result = "PROCEDURAL ERROR DETECTED"
        result_color = colors.red
    else:
        result = "NO PROCEDURAL ERROR DETECTED"
        result_color = colors.green

    result_table = Table(
        [
            [
                Paragraph(
                    f"<b>{result}</b>",
                    ParagraphStyle(
                        "Result",
                        parent=normal_style,
                        fontSize=14,
                        alignment=TA_CENTER,
                        textColor=result_color
                    )
                )
            ]
        ],
        colWidths=[7 * inch]
    )

    result_table.setStyle(
        TableStyle(
            [
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    1,
                    result_color
                ),
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.whitesmoke
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
                    15
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    15
                )
            ]
        )
    )

    story.append(result_table)
    story.append(Spacer(1, 20))

    # --------------------------------------------------
    # ANALYSIS STATISTICS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "2. Analysis Statistics",
            heading_style
        )
    )

    error_rate = 0

    if total_sequences > 0:
        error_rate = (
            error_sequences / total_sequences
        ) * 100

    correct_rate = 0

    if total_sequences > 0:
        correct_rate = (
            correct_sequences / total_sequences
        ) * 100

    statistics = [
        ["Metric", "Value"],
        ["Total Frames", str(total_frames)],
        ["Temporal Sequences", str(total_sequences)],
        ["Correct Sequences", str(correct_sequences)],
        ["Error Sequences", str(error_sequences)],
        ["Correct Detection Rate", f"{correct_rate:.2f}%"],
        ["Error Detection Rate", f"{error_rate:.2f}%"],
        ["Error Events", str(error_events)],
        ["Average Prediction Confidence", f"{avg_confidence:.2f}%"],
        ["Error Confidence", f"{error_confidence:.2f}%"]
    ]

    stats_table = Table(
        statistics,
        colWidths=[4.5 * inch, 2.5 * inch]
    )

    stats_table.setStyle(
        TableStyle(
            [
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
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#f3f4f6")
                    ]
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(stats_table)

    # --------------------------------------------------
    # DETECTED PROCEDURAL STEPS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "3. Detected Procedural Errors",
            heading_style
        )
    )

    if detected_steps:

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

        for step in detected_steps:

            step_data.append(
                [
                    str(step.get("step", "-")),
                    str(step.get("step_id", "-")),
                    f"{float(step.get('start', 0)):.2f}s",
                    f"{float(step.get('end', 0)):.2f}s",
                    str(step.get("status", "ERROR")),
                    f"{float(step.get('confidence', 0)):.2f}%"
                ]
            )

        step_table = Table(
            step_data,
            colWidths=[
                0.65 * inch,
                0.75 * inch,
                0.9 * inch,
                0.9 * inch,
                1.0 * inch,
                1.0 * inch
            ],
            repeatRows=1
        )

        step_table.setStyle(
            TableStyle(
                [
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
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.grey
                    ),
                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE"
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.white,
                            colors.HexColor("#f3f4f6")
                        ]
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
                ]
            )
        )

        story.append(step_table)

    else:

        story.append(
            Paragraph(
                "No procedural errors were detected.",
                normal_style
            )
        )

    # --------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------

    story.append(
        Paragraph(
            "4. Detection Interpretation",
            heading_style
        )
    )

    interpretation = (
        f"The CNN-LSTM temporal model analyzed {total_sequences} "
        f"temporal sequences. "
        f"{error_sequences} sequence(s) were classified as "
        f"potential procedural errors. "
        f"These temporal detections were grouped into "
        f"{error_events} continuous procedural event(s)."
    )

    story.append(
        Paragraph(
            interpretation,
            normal_style
        )
    )

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "The detected temporal regions were matched against "
            "the procedural step report to identify the affected "
            "procedure step.",
            normal_style
        )
    )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    story.append(Spacer(1, 30))

    story.append(
        Paragraph(
            "ProceVision AI | CNN-LSTM V3 | "
            "Temporal Procedural Error Detection",
            ParagraphStyle(
                "Footer",
                parent=normal_style,
                alignment=TA_CENTER,
                fontSize=8,
                textColor=colors.grey
            )
        )
    )

    doc.build(story)

    return output_path