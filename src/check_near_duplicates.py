
from pathlib import Path
from collections import defaultdict
import csv

from PIL import Image, ImageOps
import imagehash

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "metadata" / "har_labels.csv"
OUTPUT_PATH = ROOT / "metadata" / "near_duplicates.csv"

SEARCH_DIRS = [
    ROOT / "dataset_candidates",
    ROOT / "dataset_raw",
]

VALID_LABELS = {"standing", "sitting", "moving"}
EXTENSIONS = {".jpg", ".jpeg", ".png"}
THRESHOLD = 8

# Index all images
image_index = defaultdict(list)
for folder in SEARCH_DIRS:
    if folder.exists():
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                image_index[path.name].append(path)

# Read confirmed labels
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    rows = [
        row for row in csv.DictReader(f)
        if row["label"].strip().lower() in VALID_LABELS
    ]

hashes = []
missing = []
errors = []

for row in rows:
    name = row["file_name"].strip()
    paths = image_index.get(name, [])

    if not paths:
        missing.append(name)
        continue

    try:
        # Previous SHA-256 check confirmed no filename conflicts
        with Image.open(paths[0]) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            phash = imagehash.phash(img)

        hashes.append((name, row["label"].strip().lower(), phash))
    except Exception as e:
        errors.append((name, str(e)))

# Compare each pair once
pairs = []

for i in range(len(hashes)):
    name1, label1, hash1 = hashes[i]

    for j in range(i + 1, len(hashes)):
        name2, label2, hash2 = hashes[j]
        distance = hash1 - hash2

        if distance <= THRESHOLD:
            pairs.append({
                "image_1": name1,
                "label_1": label1,
                "image_2": name2,
                "label_2": label2,
                "phash_distance": distance,
            })

pairs.sort(key=lambda x: x["phash_distance"])

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_1", "label_1",
            "image_2", "label_2",
            "phash_distance"
        ]
    )
    writer.writeheader()
    writer.writerows(pairs)

print("\n===== NEAR DUPLICATE REPORT =====")
print("Expected images:", len(rows))
print("Successfully hashed:", len(hashes))
print("Missing images:", len(missing))
print("Image errors:", len(errors))
print("Potential duplicate pairs:", len(pairs))
print("Report saved:", OUTPUT_PATH)

print("\nTop 10 potential pairs:")
for pair in pairs[:10]:
    print(pair)

if missing:
    print("\nMissing:", missing[:10])

if errors:
    print("\nErrors:", errors[:10])
