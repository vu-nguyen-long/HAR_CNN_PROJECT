
from pathlib import Path
from collections import Counter
import hashlib
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_DIR / "dataset_candidates"
LABEL_FILE = PROJECT_DIR / "metadata" / "har_labels.csv"

df = pd.read_csv(LABEL_FILE, dtype={"file_name": str})

valid_labels = {"standing", "sitting", "moving"}
valid_df = df[df["label"].isin(valid_labels)].copy()

print("Tong anh da gan nhan:", len(df))
print("Anh thuoc 3 class:", len(valid_df))

missing_files = []
corrupted_files = []
duplicates = []
hashes = {}

for _, row in valid_df.iterrows():
    file_name = row["file_name"]
    image_path = IMAGE_DIR / file_name

    if not image_path.exists():
        missing_files.append(file_name)
        continue

    try:
        with Image.open(image_path) as img:
            img.verify()

        file_hash = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()

        if file_hash in hashes:
            duplicates.append(
                (file_name, hashes[file_hash])
            )
        else:
            hashes[file_hash] = file_name

    except Exception:
        corrupted_files.append(file_name)

print("\n=== KET QUA DATA CLEANING ===")
print("Anh bi thieu:", len(missing_files))
print("Anh bi loi:", len(corrupted_files))
print("Anh trung lap:", len(duplicates))

for file_name in missing_files:
    print("Missing:", file_name)

for file_name in corrupted_files:
    print("Corrupted:", file_name)

for file_name, original in duplicates:
    print("Duplicate:", file_name, "=", original)

print("\n=== PHAN BO CLASS ===")
print(valid_df["label"].value_counts())

print("\nHoan thanh kiem tra dataset!")
 