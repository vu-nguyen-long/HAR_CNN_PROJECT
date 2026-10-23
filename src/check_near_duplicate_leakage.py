
from pathlib import Path
from itertools import combinations
from PIL import Image, ImageOps
import imagehash
import csv

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
REPORT = ROOT / "metadata" / "near_duplicate_leakage.csv"

SPLITS = ("train", "val", "test")
CLASSES = ("standing", "sitting", "moving")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

THRESHOLD = 8

records = []

for split in SPLITS:
    for label in CLASSES:
        folder = DATASET / split / label

        if not folder.is_dir():
            raise RuntimeError(f"Missing folder: {folder}")

        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue

            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                phash = imagehash.phash(img.convert("RGB"))

            records.append({
                "split": split,
                "label": label,
                "file_name": path.name,
                "hash": phash,
            })

suspected_pairs = []

for a, b in combinations(records, 2):
    # Chỉ quan tâm ảnh nằm ở hai tập khác nhau
    if a["split"] == b["split"]:
        continue

    distance = a["hash"] - b["hash"]

    if distance <= THRESHOLD:
        suspected_pairs.append({
            "file_a": a["file_name"],
            "split_a": a["split"],
            "label_a": a["label"],
            "file_b": b["file_name"],
            "split_b": b["split"],
            "label_b": b["label"],
            "phash_distance": distance,
        })

REPORT.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_a", "split_a", "label_a",
            "file_b", "split_b", "label_b",
            "phash_distance",
        ],
    )
    writer.writeheader()
    writer.writerows(suspected_pairs)

print("\n===== NEAR-DUPLICATE LEAKAGE CHECK =====")
print("Total images:", len(records))
print("pHash threshold:", THRESHOLD)
print("Cross-split suspicious pairs:", len(suspected_pairs))
print("Report:", REPORT)

if not suspected_pairs:
    print("\nNEAR-DUPLICATE LEAKAGE: NOT DETECTED")
else:
    print("\nWARNING: SUSPECTED NEAR-DUPLICATES")
    for pair in suspected_pairs:
        print(
            pair["file_a"], pair["split_a"],
            "<->",
            pair["file_b"], pair["split_b"],
            "distance =", pair["phash_distance"]
        )
