
from datasets import load_dataset
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "dataset_candidates" / "batch9_standing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "metadata" / "batch9_standing.csv"

TARGET = 30

# Lay dataset
ds = load_dataset(
    "Bingsu/Human_Action_Recognition",
    split="train",
    streaming=True
)

records = []
count = 0

# Luu y: can xac minh ID cua class standing
features = ds.features["labels"]
print("Dataset labels:", features)

standing_id = features.str2int("standing")
print("Standing class ID:", standing_id)

for index, item in enumerate(ds):
    if item["labels"] != standing_id:
        continue

    filename = f"batch9_standing_{index:05d}.jpg"
    path = OUTPUT_DIR / filename

    if path.exists():
        continue

    try:
        image = item["image"].convert("RGB")
        image.save(path, quality=95)

        records.append({
            "file_name": filename,
            "label": "standing",
            "source": "Bingsu/Human_Action_Recognition",
            "source_index": index
        })

        count += 1

        if count >= TARGET:
            break

    except Exception as e:
        print("Error:", index, e)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file_name", "label", "source", "source_index"]
    )
    writer.writeheader()
    writer.writerows(records)

print("\n===== DOWNLOAD REPORT =====")
print("Downloaded:", count)
print("Saved images:", OUTPUT_DIR)
print("Saved metadata:", CSV_PATH)
