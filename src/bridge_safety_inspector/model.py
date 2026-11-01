from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


class FeatureScaler(Protocol):
    def transform(self, features: Any) -> Any: ...


class ProbabilityClassifier(Protocol):
    classes_: Any

    def predict(self, features: Any) -> Any: ...

    def predict_proba(self, features: Any) -> Any: ...


@dataclass
class TrainedCrackClassifier:
    classifier: ProbabilityClassifier
    scaler: FeatureScaler
    image_size: int

    def predict(self, paths: Sequence[Path | str]) -> list[str]:
        from bridge_safety_inspector.train import extract_features

        features = extract_features(paths, self.image_size)
        scaled_features = self.scaler.transform(features)
        return self.classifier.predict(scaled_features).tolist()

    def predict_with_confidence(
        self, paths: Sequence[Path | str]
    ) -> list[tuple[str, float]]:
        from bridge_safety_inspector.train import extract_features

        features = extract_features(paths, self.image_size)
        scaled_features = self.scaler.transform(features)
        predictions = self.classifier.predict(scaled_features)
        probabilities = self.classifier.predict_proba(scaled_features)
        class_indices = {
            label: index for index, label in enumerate(self.classifier.classes_)
        }
        return [
            (str(label), float(probabilities[row, class_indices[label]]))
            for row, label in enumerate(predictions)
        ]
