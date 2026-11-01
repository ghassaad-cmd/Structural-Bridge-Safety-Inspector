import numpy as np
from PIL import Image

from bridge_safety_inspector.dataset import DATASET_CLASSES
from bridge_safety_inspector.train import extract_features, fit_with_early_stopping


def test_extract_features_decodes_and_resizes_grayscale_image(tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("L", (330, 330), color=128).save(image_path)

    features = extract_features([image_path], image_size=64)

    assert features.shape == (1, 1568)
    assert np.isfinite(features).all()


def test_mlp_training_uses_validation_early_stopping():
    train_rng = np.random.default_rng(42)
    train_features = np.vstack(
        [
            train_rng.normal(loc=index * 3, scale=0.1, size=(8, 4))
            for index in range(len(DATASET_CLASSES))
        ]
    )
    train_labels = np.repeat(DATASET_CLASSES, 8)
    validation_rng = np.random.default_rng(43)
    validation_features = np.vstack(
        [
            validation_rng.normal(loc=index * 3, scale=0.1, size=(4, 4))
            for index in range(len(DATASET_CLASSES))
        ]
    )
    validation_labels = np.repeat(DATASET_CLASSES, 4)

    classifier, best_epoch, epochs_run, best_validation_f1 = fit_with_early_stopping(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        seed=42,
        max_epochs=2,
        patience=1,
    )

    assert 1 <= best_epoch <= epochs_run <= 2
    assert best_validation_f1 > 0
    assert set(classifier.classes_) == set(DATASET_CLASSES)
