
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"

CLASSES = ["standing", "sitting", "moving"]
EXPECTED = {
    "train": 120,
    "val": 15,
    "test": 15
}

total_images = 0
errors = []

print("=== FINAL DATASET VALIDATION ===")

for split, expected_count in EXPECTED.items():
    print(f"\n{split.upper()}")

    for label in CLASSES:
        folder = DATASET_DIR / split / label

        if not folder.is_dir():
            errors.append(f"Missing folder: {folder}")
            print(f"{label}: MISSING FOLDER")
            continue

        images = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower()
            in [".jpg", ".jpeg", ".png"]
        ]

        count = len(images)
        total_images += count

        print(f"{label}: {count} images")

        if count != expected_count:
            errors.append(
                f"{split}/{label}: expected "
                f"{expected_count}, found {count}"
            )

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception as e:
                errors.append(
                    f"Invalid image: {image_path} - {e}"
                )

print("\n=== SUMMARY ===")
print("Total images:", total_images)
print("Expected images:", 450)

if total_images != 450:
    errors.append(
        f"Expected 450 images, found {total_images}"
    )

if errors:
    print(f"VALIDATION FAILED: {len(errors)} issues")
    for error in errors:
        print("-", error)
else:
    print("VALIDATION PASSED")

# ===== SHA-256 DUPLICATE CHECK =====

import hashlib
from collections import defaultdict

print("\n=== SHA-256 DUPLICATE CHECK ===")

hash_records = defaultdict(list)

for split in EXPECTED:
    for label in CLASSES:
        folder = DATASET_DIR / split / label

        if not folder.is_dir():
            continue

        for image_path in folder.iterdir():
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in [
                ".jpg", ".jpeg", ".png"
            ]:
                continue

            sha256 = hashlib.sha256()

            with open(image_path, "rb") as f:
                for chunk in iter(
                    lambda: f.read(1024 * 1024), b""
                ):
                    sha256.update(chunk)

            hash_records[sha256.hexdigest()].append(
                (split, label, image_path.name)
            )

duplicate_groups = {
    digest: records
    for digest, records in hash_records.items()
    if len(records) > 1
}

cross_split_duplicates = 0

for digest, records in duplicate_groups.items():
    splits = {record[0] for record in records}

    if len(splits) > 1:
        cross_split_duplicates += 1
        print("Cross-split duplicate:", records)

print("Unique SHA-256:", len(hash_records))
print("Duplicate groups:", len(duplicate_groups))
print("Cross-split duplicate groups:", cross_split_duplicates)

if duplicate_groups:
    print("SHA-256 CHECK: REVIEW REQUIRED")
else:
    print("SHA-256 CHECK: PASSED")

# ===== METADATA CONSISTENCY CHECK =====

import pandas as pd

print("\n=== METADATA CONSISTENCY CHECK ===")

manifest_path = PROJECT_DIR / "metadata" / "dataset_split_manifest.csv"
manifest = pd.read_csv(manifest_path)

required_columns = {
    "file_name", "label", "split", "output_path"
}

missing_columns = required_columns - set(manifest.columns)

if missing_columns:
    raise ValueError(
        f"Missing manifest columns: {missing_columns}"
    )

metadata_errors = []

if len(manifest) != 450:
    metadata_errors.append(
        f"Expected 450 manifest rows, found {len(manifest)}"
    )

expected_paths = set()

for _, row in manifest.iterrows():
    split = str(row["split"])
    label = str(row["label"])
    file_name = str(row["file_name"])

    expected_path = DATASET_DIR / split / label / file_name
    expected_paths.add(expected_path.resolve())

    if split not in EXPECTED or label not in CLASSES:
        metadata_errors.append(
            f"Invalid split/label: {split}/{label}/{file_name}"
        )
        continue

    if not expected_path.is_file():
        metadata_errors.append(
            f"Missing image: {expected_path}"
        )

    manifest_output = PROJECT_DIR / str(row["output_path"])

    if manifest_output.resolve() != expected_path.resolve():
        metadata_errors.append(
            f"Path mismatch: {file_name}"
        )

actual_paths = {
    p.resolve()
    for split in EXPECTED
    for label in CLASSES
    for p in (DATASET_DIR / split / label).glob("*")
    if p.is_file()
    and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
}

extra_files = actual_paths - expected_paths

for path in extra_files:
    metadata_errors.append(f"Unlisted image: {path}")

print("Manifest rows:", len(manifest))
print("Actual image files:", len(actual_paths))
print("Metadata errors:", len(metadata_errors))

if metadata_errors:
    print("METADATA CHECK: FAILED")
    for error in metadata_errors[:20]:
        print("-", error)
else:
    print("METADATA CHECK: PASSED")

# ===== PREPROCESSING VALIDATION =====

import numpy as np
from preprocessing import preprocess_image

print("\n=== PREPROCESSING VALIDATION ===")

preprocessing_errors = []
processed_count = 0

for split in EXPECTED:
    for label in CLASSES:
        folder = DATASET_DIR / split / label

        for image_path in folder.iterdir():
            if (
                not image_path.is_file()
                or image_path.suffix.lower()
                not in [".jpg", ".jpeg", ".png"]
            ):
                continue

            try:
                image = preprocess_image(
                    image_path,
                    normalize=True
                )

                if image.shape != (224, 224, 3):
                    raise ValueError(
                        f"Invalid shape: {image.shape}"
                    )

                if image.dtype != np.float32:
                    raise ValueError(
                        f"Invalid dtype: {image.dtype}"
                    )

                if not np.isfinite(image).all():
                    raise ValueError(
                        "Image contains NaN or Inf"
                    )

                if image.min() < 0 or image.max() > 1:
                    raise ValueError(
                        "Pixel values outside [0,1]"
                    )

                processed_count += 1

            except Exception as e:
                preprocessing_errors.append(
                    f"{image_path.name}: {e}"
                )

print("Successfully processed:", processed_count)
print("Preprocessing errors:", len(preprocessing_errors))

if processed_count == 450 and not preprocessing_errors:
    print("PREPROCESSING CHECK: PASSED")
else:
    print("PREPROCESSING CHECK: FAILED")
    for error in preprocessing_errors[:20]:
        print("-", error)
