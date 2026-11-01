from pathlib import Path

import joblib
import pytest

from bridge_safety_inspector.dataset import DATASET_CLASSES
from bridge_safety_inspector.model import TrainedCrackClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "cics_hog_mlp.joblib"


@pytest.fixture(scope="module")
def trained_model() -> TrainedCrackClassifier:
    if not MODEL_PATH.is_file():
        pytest.skip("The trained model artifact is not present.")
    return joblib.load(MODEL_PATH)


@pytest.mark.parametrize("category", DATASET_CLASSES)
def test_trained_model_predicts_images_from_each_dataset_category(
    trained_model: TrainedCrackClassifier, category: str
):
    category_directory = PROJECT_ROOT / "data" / "CICS" / category
    image_paths = sorted(category_directory.glob("*.jpg"))
    if not image_paths:
        pytest.skip(f"No dataset images are available for {category}.")

    results = trained_model.predict_with_confidence([image_paths[0]])

    assert len(results) == 1
    predicted_category, confidence = results[0]
    assert predicted_category in DATASET_CLASSES
    assert 0.0 <= confidence <= 1.0
