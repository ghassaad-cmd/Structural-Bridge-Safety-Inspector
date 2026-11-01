import pytest

from bridge_safety_inspector.health import (
    DEMO_CONFIDENCE_THRESHOLD,
    BridgeInspectionResult,
    assess_crack_classification,
    determine_priority,
)


def test_bridge_inspection_result_calculates_severity():
    result = BridgeInspectionResult(
        bridge_id="B-101",
        crack_density=0.85,
        corrosion_score=0.9,
        defect_count=12,
        condition_rating=2,
    )

    assert result.overall_severity >= 0.8
    assert result.severity_label == "Critical"


def test_priority_ranking_places_critical_items_first():
    results = [
        BridgeInspectionResult("B-102", 0.35, 0.4, 3, 4),
        BridgeInspectionResult("B-103", 0.8, 0.9, 10, 2),
        BridgeInspectionResult("B-104", 0.5, 0.5, 5, 3),
    ]

    ranked = determine_priority(results)

    assert ranked[0].bridge_id == "B-103"
    assert [item.bridge_id for item in ranked[:2]] == ["B-103", "B-104"]


@pytest.mark.parametrize(
    ("category", "severity", "priority"),
    [
        ("Without Crack", "Low (preliminary)", "Routine review"),
        ("Simple Cracks", "Moderate (preliminary)", "Scheduled review"),
        ("Multibranched Crack", "High (preliminary)", "Prompt review"),
    ],
)
def test_classification_assessment_maps_each_category(category, severity, priority):
    assessment = assess_crack_classification(category, confidence=0.9)

    assert assessment.predicted_category == category
    assert assessment.confidence == 0.9
    assert assessment.severity_level == severity
    assert assessment.maintenance_priority == priority
    assert assessment.explanation
    assert "not an engineering assessment" in assessment.explanation or (
        "not confirmation" in assessment.explanation
    )
    assert assessment.manual_inspection_required is False


@pytest.mark.parametrize(
    "category",
    ["Without Crack", "Simple Cracks", "Multibranched Crack"],
)
def test_low_confidence_requires_manual_inspection(category):
    assessment = assess_crack_classification(
        category,
        confidence=DEMO_CONFIDENCE_THRESHOLD - 0.01,
    )

    assert assessment.predicted_category == category
    assert assessment.severity_level == "Uncertain"
    assert assessment.maintenance_priority == "Manual inspection"
    assert assessment.manual_inspection_required is True
    assert "Do not rely on this classification" in assessment.explanation


def test_confidence_at_demo_threshold_uses_indicative_category_assessment():
    assessment = assess_crack_classification(
        "Simple Cracks",
        confidence=DEMO_CONFIDENCE_THRESHOLD,
    )

    assert assessment.severity_level == "Moderate (preliminary)"
    assert assessment.manual_inspection_required is False


@pytest.mark.parametrize(
    ("category", "confidence"),
    [("Unknown", 0.9), ("Simple Cracks", -0.1), ("Simple Cracks", 1.1)],
)
def test_classification_assessment_rejects_invalid_input(category, confidence):
    with pytest.raises(ValueError):
        assess_crack_classification(category, confidence)
