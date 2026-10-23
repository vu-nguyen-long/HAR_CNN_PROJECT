
from pathlib import Path
from PIL import Image
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "metadata" / "har_labels.csv"

SEARCH_DIRS = [
    PROJECT_DIR / "dataset_candidates",
    PROJECT_DIR / "dataset_raw",
]

VALID_LABELS = {"standing", "sitting", "moving"}

# Lập danh sách đường dẫn ảnh theo tên file
image_paths = {}

for folder in SEARCH_DIRS:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }:
            image_paths.setdefault(path.name, []).append(path)

valid = 0
missing = []
corrupted = []
ambiguous = []

with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["label"].strip().lower() not in VALID_LABELS:
            continue

        filename = row["file_name"].strip()
        paths = image_paths.get(filename, [])

        if not paths:
            missing.append(filename)
            continue

        if len(paths) > 1:
            ambiguous.append(filename)
            continue

        try:
            with Image.open(paths[0]) as img:
                img.verify()
            valid += 1
        except Exception:
            corrupted.append(filename)

print("\n===== IMAGE INTEGRITY REPORT =====")
print("Valid images:", valid)
print("Missing images:", len(missing))
print("Corrupted images:", len(corrupted))
print("Multiple file matches:", len(ambiguous))

if missing:
    print("Missing:", missing[:10])

if corrupted:
    print("Corrupted:", corrupted[:10])

if ambiguous:
    print("Multiple matches:", ambiguous[:10])

