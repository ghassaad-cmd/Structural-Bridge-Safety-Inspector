from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time
from typing import Sequence

from bridge_safety_inspector.dataset import DATASET_CLASSES, DatasetSplits, ImageSample, create_splits
from bridge_safety_inspector.model import TrainedCrackClassifier


def extract_features(paths: Sequence[Path | str], image_size: int):
    import numpy as np
    from PIL import Image, UnidentifiedImageError
    from skimage.feature import hog

    features = []
    for path in paths:
        image_path = Path(path)
        try:
            with Image.open(image_path) as image:
                grayscale = image.convert("L").resize((image_size, image_size))
                pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as error:
            raise ValueError(f"Unable to decode dataset image {image_path}: {error}") from error
        features.append(
            hog(
                pixels,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
        )
    return np.asarray(features, dtype=np.float32)


def _balanced_indices(labels, rng: random.Random):
    import numpy as np

    indices_by_class = {
        label: np.flatnonzero(labels == label) for label in DATASET_CLASSES
    }
    if any(len(indices) == 0 for indices in indices_by_class.values()):
        raise ValueError("The training split must contain samples from every class.")

    target_size = max(len(indices) for indices in indices_by_class.values())
    indices = np.concatenate(
        [
            np.asarray(rng.choices(indices.tolist(), k=target_size), dtype=np.int64)
            for indices in indices_by_class.values()
        ]
    )
    rng.shuffle(indices)
    return indices


def fit_with_early_stopping(
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
):
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.neural_network import MLPClassifier

    batch_size = min(64, len(train_features))
    classifier = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        batch_size=batch_size,
        learning_rate_init=0.001,
        max_iter=1,
        random_state=seed,
    )
    rng = random.Random(seed)
    best_f1 = -1.0
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(1, max_epochs + 1):
        indices = _balanced_indices(train_labels, rng)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            classifier.set_params(batch_size=len(batch_indices))
            classifier.partial_fit(
                train_features[batch_indices],
                train_labels[batch_indices],
                classes=np.asarray(DATASET_CLASSES) if epoch == 1 and start == 0 else None,
            )

        epochs_run = epoch
        validation_predictions = classifier.predict(validation_features)
        validation_f1 = f1_score(
            validation_labels,
            validation_predictions,
            labels=DATASET_CLASSES,
            average="macro",
            zero_division=0,
        )
        print(
            f"Epoch {epoch}/{max_epochs}: "
            f"loss={classifier.loss_:.4f}, validation_macro_f1={validation_f1:.4f}",
            flush=True,
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            best_weights = (deepcopy(classifier.coefs_), deepcopy(classifier.intercepts_))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_weights is None:
        raise RuntimeError("Training completed without a checkpoint to restore.")
    classifier.coefs_, classifier.intercepts_ = best_weights
    return classifier, best_epoch, epochs_run, best_f1


def _labels(samples: Sequence[ImageSample]):
    import numpy as np

    return np.asarray([sample.label for sample in samples])


def _evaluate(classifier, scaler, samples: Sequence[ImageSample], image_size: int):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    features = extract_features([sample.path for sample in samples], image_size)
    labels = _labels(samples)
    predictions = classifier.predict(scaler.transform(features))
    return {
        "sample_count": len(samples),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=DATASET_CLASSES, average="macro", zero_division=0)
        ),
        "per_class": classification_report(
            labels,
            predictions,
            labels=DATASET_CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=DATASET_CLASSES
        ).tolist(),
    }


def train(
    dataset_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    image_size: int = 64,
    max_epochs: int = 12,
    patience: int = 3,
) -> dict:
    import joblib
    from sklearn.preprocessing import StandardScaler

    if image_size < 32 or image_size % 8:
        raise ValueError("Image size must be at least 32 and divisible by 8.")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive.")

    run_started = time.perf_counter()
    splits: DatasetSplits = create_splits(dataset_dir, seed=seed)
    for split_name, samples in (
        ("training", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        missing_labels = set(DATASET_CLASSES) - {sample.label for sample in samples}
        if missing_labels:
            raise ValueError(f"The {split_name} split is missing classes: {sorted(missing_labels)}")

    train_features = extract_features([sample.path for sample in splits.train], image_size)
    validation_features = extract_features(
        [sample.path for sample in splits.validation], image_size
    )
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    validation_features = scaler.transform(validation_features)
    train_labels = _labels(splits.train)
    validation_labels = _labels(splits.validation)

    fitting_started = time.perf_counter()
    classifier, best_epoch, epochs_run, best_validation_f1 = fit_with_early_stopping(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
    )
    fitting_seconds = time.perf_counter() - fitting_started

    validation_metrics = _evaluate(classifier, scaler, splits.validation, image_size)
    test_metrics = _evaluate(classifier, scaler, splits.test, image_size)
    model = TrainedCrackClassifier(classifier, scaler, image_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cics_hog_mlp.joblib"
    metrics_path = output_dir / "cics_hog_mlp.metrics.json"
    joblib.dump(model, model_path, compress=3)

    results = {
        "dataset_dir": str(dataset_dir),
        "classes": list(DATASET_CLASSES),
        "seed": seed,
        "split_counts": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
            "excluded_conflicting_label_files": len(splits.excluded_conflicts),
        },
        "image_size": image_size,
        "feature_extractor": "8-orientation HOG, 8x8 pixels per cell, 2x2 cells per block",
        "classifier": "one-hidden-layer MLP (64 units), balanced mini-batches",
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_validation_macro_f1_during_training": best_validation_f1,
        "validation": validation_metrics,
        "test": test_metrics,
        "fitting_seconds": fitting_seconds,
        "model_path": str(model_path),
    }
    results["total_seconds"] = time.perf_counter() - run_started
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved trained model: {model_path}", flush=True)
    print(f"Saved evaluation metrics: {metrics_path}", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the CICS crack classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/CICS"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    train(
        args.dataset_dir,
        args.output_dir,
        seed=args.seed,
        image_size=args.image_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
