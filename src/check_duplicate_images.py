
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "metadata" / "har_labels.csv"
SEARCH_DIRS = [
    ROOT / "dataset_candidates",
    ROOT / "dataset_raw",
]

VALID_LABELS = {"standing", "sitting", "moving"}
EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Tim tat ca file anh
image_index = defaultdict(list)

for folder in SEARCH_DIRS:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            image_index[path.name].append(path)

# Doc danh sach anh da gan nhan
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    rows = [
        row for row in csv.DictReader(f)
        if row["label"].strip().lower() in VALID_LABELS
    ]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

hash_groups = defaultdict(list)
missing = []
conflicts = []

for row in rows:
    name = row["file_name"].strip()
    paths = image_index.get(name, [])

    if not paths:
        missing.append(name)
        continue

    hashes = {sha256(p) for p in paths}

    if len(hashes) > 1:
        conflicts.append(name)
        continue

    file_hash = next(iter(hashes))
    hash_groups[file_hash].append((name, row["label"]))

duplicate_groups = [
    group for group in hash_groups.values()
    if len(group) > 1
]

print("\n===== DUPLICATE REPORT =====")
print("Labeled records:", len(rows))
print("Unique image contents:", len(hash_groups))
print("Duplicate groups:", len(duplicate_groups))
print("Missing files:", len(missing))
print("Filename conflicts:", len(conflicts))

for group in duplicate_groups[:20]:
    print("DUPLICATE:", group)
