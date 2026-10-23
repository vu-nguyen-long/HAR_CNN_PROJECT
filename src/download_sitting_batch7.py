
from datasets import load_dataset
from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "dataset_candidates" / "batch7_sitting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "Bingsu/Human_Action_Recognition",
    split="train",
    streaming=True
)

count = 0
records = []

for index, item in enumerate(ds):
    if item["labels"] != 11:
        continue

    filename = f"batch7_sitting_{index:05d}.jpg"
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

    if count >= 100:
        break

csv_path = PROJECT_DIR / "metadata" / "batch7_sitting.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file_name", "label", "source", "source_index"
    ])
    writer.writeheader()
    writer.writerows(records)

print("Da tai:", count, "anh sitting")
print("Thu muc:", OUTPUT_DIR)
print("CSV:", csv_path)
