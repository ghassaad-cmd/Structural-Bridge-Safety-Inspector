from __future__ import annotations

from datetime import datetime

from bridge_safety_inspector.health import CrackClassificationAssessment


def build_inspection_report(
    *,
    image_filename: str,
    predicted_category: str,
    confidence: float,
    assessment: CrackClassificationAssessment,
    inspected_at: datetime | None = None,
) -> str:
    """Build a downloadable Markdown summary for one prototype inspection."""
    inspection_time = inspected_at or datetime.now().astimezone()
    timestamp = inspection_time.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    return "\n".join(
        [
            "# Automated Structural Bridge Safety Inspector",
            "",
            "## Inspection result",
            f"- **Image filename:** {image_filename}",
            f"- **Predicted crack category:** {predicted_category}",
            f"- **Prediction confidence:** {confidence:.1%}",
            f"- **Preliminary severity:** {assessment.severity_level}",
            f"- **Maintenance priority:** {assessment.maintenance_priority}",
            f"- **Inspection date and time:** {timestamp}",
            "",
            "## Assessment explanation",
            assessment.explanation,
            "",
            "## Important limitations and warnings",
            "- This is a prototype and is not a certified engineering safety assessment.",
            "- Image classification is limited to the three concrete-crack categories; it does not detect corrosion.",
            "- Results are preliminary and may be incorrect; a qualified inspector must verify observations and determine any action.",
            "- Do not use this report alone to determine bridge condition, safety, or maintenance needs.",
            "",
        ]
    )
