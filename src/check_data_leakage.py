
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
REPORT = ROOT / "metadata" / "data_leakage_report.csv"

SPLITS = ("train", "val", "test")
CLASSES = ("standing", "sitting", "moving")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

hash_groups = defaultdict(list)
counts = Counter()

for split in SPLITS:
    for label in CLASSES:
        folder = DATASET / split / label

        if not folder.is_dir():
            raise RuntimeError(f"Missing folder: {folder}")

        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in EXTENSIONS:
                continue

            file_hash = sha256_file(path)

            hash_groups[file_hash].append({
                "split": split,
                "label": label,
                "file_name": path.name,
                "path": str(path.relative_to(ROOT)),
            })

            counts[(split, label)] += 1

leakage = []

for file_hash, records in hash_groups.items():
    splits_found = {r["split"] for r in records}

    if len(splits_found) > 1:
        for record in records:
            leakage.append({
                "sha256": file_hash,
                **record,
            })

REPORT.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sha256", "split", "label",
            "file_name", "path"
        ],
    )
    writer.writeheader()
    writer.writerows(leakage)

print("\n===== DATA LEAKAGE CHECK =====")

for split in SPLITS:
    total = sum(counts[(split, label)] for label in CLASSES)
    print(f"{split}: {total} images")

print("\nTotal images:", sum(counts.values()))
print("Unique SHA256:", len(hash_groups))
print("Cross-split duplicate records:", len(leakage))
print("Report:", REPORT)

if not leakage:
    print("\nEXACT DUPLICATE LEAKAGE: NOT DETECTED")
else:
    print("\nWARNING: CROSS-SPLIT DUPLICATES FOUND")
    for item in leakage:
        print(item["split"], item["path"])
