import argparse
from pathlib import Path

import numpy as np


EXPECTED_CLASSES = 40
EXPECTED_TRAIN_SAMPLES = 9843
EXPECTED_TEST_SAMPLES = 2468
MIN_POINTS = 1024


def read_nonempty_lines(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def class_name(sample_id):
    return sample_id.rsplit("_", 1)[0]


def validate_point_file(path):
    points = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise AssertionError(
            f"{path} must have shape [N, 6] for x,y,z,nx,ny,nz; got {points.shape}"
        )
    if len(points) < MIN_POINTS:
        raise AssertionError(
            f"{path} has {len(points)} points; at least {MIN_POINTS} are required"
        )
    if not np.isfinite(points).all():
        raise AssertionError(f"{path} contains NaN or infinity")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing modelnet40_train_data and modelnet40_test_data.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Validate every point file instead of one sample per class.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    train_root = root / "modelnet40_train_data" / "modelnet40_normal_resampled"
    test_root = root / "modelnet40_test_data" / "modelnet40_normal_resampled" / "test"

    names = read_nonempty_lines(train_root / "modelnet40_shape_names.txt")
    train_ids = read_nonempty_lines(train_root / "modelnet40_train.txt")
    train_files = [
        train_root / class_name(sample_id) / f"{sample_id}.txt"
        for sample_id in train_ids
    ]
    test_files = [
        path
        for name in names
        for path in sorted((test_root / name).glob("*.txt"))
    ]

    assert len(names) == EXPECTED_CLASSES, (
        f"expected {EXPECTED_CLASSES} classes, found {len(names)}"
    )
    assert len(set(names)) == len(names), "class list contains duplicate names"
    assert len(train_ids) == EXPECTED_TRAIN_SAMPLES, (
        f"expected {EXPECTED_TRAIN_SAMPLES} training samples, found {len(train_ids)}"
    )
    assert len(test_files) == EXPECTED_TEST_SAMPLES, (
        f"expected {EXPECTED_TEST_SAMPLES} test samples, found {len(test_files)}"
    )
    assert all(path.is_file() for path in train_files), "one or more train files are missing"
    assert all((test_root / name).is_dir() for name in names), (
        "one or more test class directories are missing"
    )
    assert set(train_ids).isdisjoint(path.stem for path in test_files), (
        "train and test sample identifiers overlap"
    )

    if args.full:
        files_to_check = train_files + test_files
    else:
        first_train_by_class = {}
        for path in train_files:
            first_train_by_class.setdefault(path.parent.name, path)
        first_test_by_class = {}
        for path in test_files:
            first_test_by_class.setdefault(path.parent.name, path)
        files_to_check = list(first_train_by_class.values()) + list(
            first_test_by_class.values()
        )

    for path in files_to_check:
        validate_point_file(path)

    val_count = sum(
        max(1, round(sum(class_name(sample_id) == name for sample_id in train_ids) * 0.15))
        for name in names
    )
    print(
        f"classes={len(names)} train_total={len(train_ids)} "
        f"train_split={len(train_ids) - val_count} val_split={val_count} "
        f"test={len(test_files)}"
    )
    print(f"validated_point_files={len(files_to_check)} full_check={args.full}")
    print("dataset layout and x,y,z,nx,ny,nz format are valid")


if __name__ == "__main__":
    main()
