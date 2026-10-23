
from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import shutil

ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = ROOT / "metadata" / "har_labels.csv"
OUTPUT_DIR = ROOT / "dataset_final"

SEARCH_DIRS = [
    ROOT / "dataset_candidates",
    ROOT / "dataset_raw",
]

CLASSES = ("standing", "sitting", "moving")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Step 1: Read confirmed labels
with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    rows = [
        row for row in csv.DictReader(f)
        if row["label"].strip().lower() in CLASSES
    ]

# Step 2: Build filename index
image_index = defaultdict(list)

for folder in SEARCH_DIRS:
    if folder.exists():
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:
                image_index[path.name].append(path)

# Step 3: Validate before copying
errors = []
copy_plan = []
seen_names = set()

for row in rows:
    name = row["file_name"].strip()
    label = row["label"].strip().lower()
    paths = image_index.get(name, [])

    if name in seen_names:
        errors.append(f"Duplicate CSV filename: {name}")
        continue
    seen_names.add(name)

    if not paths:
        errors.append(f"Missing image: {name}")
        continue

    if len(paths) > 1:
        hashes = {sha256(p) for p in paths}
        if len(hashes) > 1:
            errors.append(f"Filename conflict: {name}")
            continue

    copy_plan.append((paths[0], OUTPUT_DIR / label / name))

counts = Counter(row["label"].strip().lower() for row in rows)

if len(rows) != 450:
    errors.append(f"Expected 450 labels, found {len(rows)}")

for label in CLASSES:
    if counts[label] != 150:
        errors.append(f"{label}: expected 150, found {counts[label]}")

if errors:
    print("VALIDATION FAILED:")
    for error in errors:
        print("-", error)
    raise SystemExit(1)

# Step 4: Copy safely
copied = 0
already_present = 0

for src, dst in copy_plan:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if sha256(src) != sha256(dst):
            raise RuntimeError(
                f"Destination file differs: {dst}"
            )
        already_present += 1
    else:
        shutil.copy2(src, dst)
        copied += 1

# Step 5: Verify output
print("\n===== DATASET FINAL REPORT =====")

for label in CLASSES:
    folder = OUTPUT_DIR / label
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    print(f"{label}: {len(files)} images")

print("Expected images:", len(rows))
print("Newly copied:", copied)
print("Already present:", already_present)
print("Output:", OUTPUT_DIR)
