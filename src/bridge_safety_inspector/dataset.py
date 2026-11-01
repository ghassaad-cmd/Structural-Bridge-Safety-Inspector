from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import random


DATASET_CLASSES = ("Multibranched Crack", "Simple Cracks", "Without Crack")


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: str
    sha256: str


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[ImageSample, ...]
    validation: tuple[ImageSample, ...]
    test: tuple[ImageSample, ...]
    excluded_conflicts: tuple[ImageSample, ...]


def create_splits(dataset_dir: Path | str, seed: int = 42) -> DatasetSplits:
    """Create reproducible 70/15/15 splits without separating exact duplicates."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    class_directories = {entry.name for entry in root.iterdir() if entry.is_dir()}
    missing_classes = set(DATASET_CLASSES) - class_directories
    unexpected_classes = class_directories - set(DATASET_CLASSES)
    if missing_classes or unexpected_classes:
        raise ValueError(
            "Dataset class directories do not match the expected classes "
            f"(missing={sorted(missing_classes)}, unexpected={sorted(unexpected_classes)})"
        )

    samples_by_digest: dict[str, list[ImageSample]] = defaultdict(list)
    for label in DATASET_CLASSES:
        paths = sorted(
            path
            for path in (root / label).iterdir()
            if path.is_file() and path.suffix.lower() == ".jpg"
        )
        if not paths:
            raise ValueError(f"No JPEG images found in {root / label}")
        for path in paths:
            digest = sha256(path.read_bytes()).hexdigest()
            samples_by_digest[digest].append(ImageSample(path, label, digest))

    unique_by_class: dict[str, list[ImageSample]] = {
        label: [] for label in DATASET_CLASSES
    }
    excluded_conflicts = []
    for samples in samples_by_digest.values():
        labels = {sample.label for sample in samples}
        if len(labels) > 1:
            excluded_conflicts.extend(samples)
            continue
        unique_by_class[samples[0].label].append(
            min(samples, key=lambda sample: sample.path.as_posix())
        )

    rng = random.Random(seed)
    train = []
    validation = []
    test = []
    for label in DATASET_CLASSES:
        samples = sorted(unique_by_class[label], key=lambda sample: sample.sha256)
        rng.shuffle(samples)

        train_end = int(len(samples) * 0.70)
        validation_end = train_end + int(len(samples) * 0.15)
        train.extend(samples[:train_end])
        validation.extend(samples[train_end:validation_end])
        test.extend(samples[validation_end:])

    return DatasetSplits(
        train=tuple(train),
        validation=tuple(validation),
        test=tuple(test),
        excluded_conflicts=tuple(
            sorted(excluded_conflicts, key=lambda sample: sample.path.as_posix())
        ),
    )
