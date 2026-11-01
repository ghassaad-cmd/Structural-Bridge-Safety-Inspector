from datetime import datetime, timezone

from bridge_safety_inspector.health import assess_crack_classification
from bridge_safety_inspector.report import build_inspection_report


def test_report_includes_inspection_result_timestamp_and_limitations():
    assessment = assess_crack_classification("Simple Cracks", 0.82)
    report = build_inspection_report(
        image_filename="bridge-deck.jpg",
        predicted_category="Simple Cracks",
        confidence=0.82,
        assessment=assessment,
        inspected_at=datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc),
    )

    assert "bridge-deck.jpg" in report
    assert "Simple Cracks" in report
    assert "82.0%" in report
    assert "Moderate (preliminary)" in report
    assert "Scheduled review" in report
    assert assessment.explanation in report
    assert "2026-11-01 09:00:00 UTC" in report
    assert "not a certified engineering safety assessment" in report
    assert "does not detect corrosion" in report
