from pathlib import Path

import pytest

from bridge_safety_inspector.dataset import DATASET_CLASSES, create_splits


def _create_dataset(root: Path, images_per_class: int = 20) -> None:
    for label in DATASET_CLASSES:
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for index in range(images_per_class):
            (class_dir / f"{index}.jpg").write_bytes(f"{label}:{index}".encode())


def test_create_splits_are_reproducible_and_keep_exact_duplicates_together(tmp_path):
    _create_dataset(tmp_path)
    first_class_dir = tmp_path / DATASET_CLASSES[0]
    (first_class_dir / "duplicate.jpg").write_bytes(
        (first_class_dir / "0.jpg").read_bytes()
    )

    first = create_splits(tmp_path, seed=7)
    second = create_splits(tmp_path, seed=7)

    assert first == second
    assert len(first.train) + len(first.validation) + len(first.test) == 60
    assert not first.excluded_conflicts
    hashes_by_split = [
        {sample.sha256 for sample in split}
        for split in (first.train, first.validation, first.test)
    ]
    assert hashes_by_split[0].isdisjoint(hashes_by_split[1])
    assert hashes_by_split[0].isdisjoint(hashes_by_split[2])
    assert hashes_by_split[1].isdisjoint(hashes_by_split[2])


def test_create_splits_excludes_exact_duplicates_with_conflicting_labels(tmp_path):
    _create_dataset(tmp_path, images_per_class=3)
    duplicate_content = b"same image, different labels"
    simple = tmp_path / "Simple Cracks" / "conflict.jpg"
    without_crack = tmp_path / "Without Crack" / "conflict.jpg"
    simple.write_bytes(duplicate_content)
    without_crack.write_bytes(duplicate_content)

    splits = create_splits(tmp_path)

    assert {sample.path for sample in splits.excluded_conflicts} == {
        simple,
        without_crack,
    }
    assert all(
        sample.sha256 != splits.excluded_conflicts[0].sha256
        for split in (splits.train, splits.validation, splits.test)
        for sample in split
    )
    assert simple.exists()
    assert without_crack.exists()


def test_create_splits_requires_all_expected_classes(tmp_path):
    for label in DATASET_CLASSES[:-1]:
        (tmp_path / label).mkdir()

    with pytest.raises(ValueError, match="missing=.*Without Crack"):
        create_splits(tmp_path)
