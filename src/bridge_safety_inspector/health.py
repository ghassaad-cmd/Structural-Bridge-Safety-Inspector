from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


DEMO_CONFIDENCE_THRESHOLD = 0.6

_CLASSIFICATION_ASSESSMENTS = {
    "Without Crack": (
        "Low (preliminary)",
        "Routine review",
        "The image was classified as Without Crack. This is not confirmation "
        "that the bridge is safe; continue normal qualified inspections.",
    ),
    "Simple Cracks": (
        "Moderate (preliminary)",
        "Scheduled review",
        "The image was classified as Simple Cracks. Arrange a qualified review "
        "of the observed area; this demonstration rule is not an engineering assessment.",
    ),
    "Multibranched Crack": (
        "High (preliminary)",
        "Prompt review",
        "The image was classified as Multibranched Crack. Prioritize qualified "
        "review of the observed area; this demonstration rule is not an engineering assessment.",
    ),
}


@dataclass(frozen=True)
class CrackClassificationAssessment:
    predicted_category: str
    confidence: float
    severity_level: str
    maintenance_priority: str
    explanation: str
    manual_inspection_required: bool


def assess_crack_classification(
    predicted_category: str,
    confidence: float,
    *,
    confidence_threshold: float = DEMO_CONFIDENCE_THRESHOLD,
) -> CrackClassificationAssessment:
    """Map a crack class to preliminary demo guidance, not a safety assessment."""
    if predicted_category not in _CLASSIFICATION_ASSESSMENTS:
        raise ValueError(f"Unsupported crack classification: {predicted_category!r}")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be a finite value between 0 and 1.")
    if (
        not math.isfinite(confidence_threshold)
        or not 0.0 <= confidence_threshold <= 1.0
    ):
        raise ValueError("confidence_threshold must be a finite value between 0 and 1.")

    if confidence < confidence_threshold:
        return CrackClassificationAssessment(
            predicted_category=predicted_category,
            confidence=confidence,
            severity_level="Uncertain",
            maintenance_priority="Manual inspection",
            explanation=(
                f"Prediction confidence ({confidence:.1%}) is below the "
                f"preliminary demonstration threshold ({confidence_threshold:.1%}). "
                "Do not rely on this classification; request manual inspection. "
                "This is not a safety assessment."
            ),
            manual_inspection_required=True,
        )

    severity_level, maintenance_priority, explanation = (
        _CLASSIFICATION_ASSESSMENTS[predicted_category]
    )
    return CrackClassificationAssessment(
        predicted_category=predicted_category,
        confidence=confidence,
        severity_level=severity_level,
        maintenance_priority=maintenance_priority,
        explanation=explanation,
        manual_inspection_required=False,
    )


@dataclass
class BridgeInspectionResult:
    bridge_id: str
    crack_density: float
    corrosion_score: float
    defect_count: int
    condition_rating: int

    @property
    def overall_severity(self) -> float:
        normalized_defects = min(self.defect_count / 15.0, 1.0)
        condition_factor = max((5 - self.condition_rating) / 4.0, 0.0)
        score = (
            0.5 * self.crack_density
            + 0.3 * self.corrosion_score
            + 0.15 * normalized_defects
            + 0.05 * condition_factor
        )
        return round(min(max(score, 0.0), 1.0), 3)

    @property
    def severity_label(self) -> str:
        if self.overall_severity >= 0.8:
            return "Critical"
        if self.overall_severity >= 0.6:
            return "High"
        if self.overall_severity >= 0.4:
            return "Moderate"
        return "Low"

    def risk_summary(self) -> str:
        return (
            f"{self.bridge_id}: {self.severity_label} risk "
            f"(severity={self.overall_severity:.2f}, defects={self.defect_count})"
        )


def determine_priority(results: Sequence[BridgeInspectionResult]) -> list[BridgeInspectionResult]:
    """Return the most urgent bridge inspections first."""
    return sorted(
        results,
        key=lambda item: (
            -item.overall_severity,
            item.condition_rating,
            -item.defect_count,
            item.bridge_id,
        ),
    )
