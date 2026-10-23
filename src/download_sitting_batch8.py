
from datasets import load_dataset
from pathlib import Path
import csv

# 1. Duong dan du an
PROJECT_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_DIR / "dataset_candidates" / "batch8_sitting"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH7_CSV = (
    PROJECT_DIR / "metadata" / "batch7_sitting.csv"
)

OUTPUT_CSV = (
    PROJECT_DIR / "metadata" / "batch8_sitting.csv"
)

# 2. Doc danh sach anh da tai Batch 7
used_indices = set()

if BATCH7_CSV.exists():
    with open(BATCH7_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            used_indices.add(int(row["source_index"]))

print("So anh da tai Batch 7:", len(used_indices))

# 3. Doc dataset Hugging Face
ds = load_dataset(
    "Bingsu/Human_Action_Recognition",
    split="train",
    streaming=True
)

# 4. Tai 120 anh Sitting moi
TARGET = 120
count = 0
records = []

for index, item in enumerate(ds):

    if item["labels"] != 11:
        continue

    # Bo qua anh da tai Batch 7
    if index in used_indices:
        continue

    filename = f"batch8_sitting_{index:05d}.jpg"
    path = OUTPUT_DIR / filename

    image = item["image"].convert("RGB")
    image.save(path, quality=95)

    records.append({
        "file_name": filename,
        "label": "sitting",
        "source": "Bingsu/Human_Action_Recognition",
        "source_index": index
    })

    count += 1

    if count >= TARGET:
        break

# 5. Luu metadata Batch 8
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name",
            "label",
            "source",
            "source_index"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

# 6. Thong ke
print("\n=== KET QUA BATCH 8 ===")
print("Anh da tai Batch 7:", len(used_indices))
print("Anh Sitting moi:", count)
print("Thu muc:", OUTPUT_DIR)
print("CSV:", OUTPUT_CSV)
