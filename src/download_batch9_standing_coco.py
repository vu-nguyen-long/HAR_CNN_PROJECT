
from pathlib import Path
import csv
import requests
from PIL import Image
from io import BytesIO

ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = ROOT / "metadata" / "batch9_standing_candidates.csv"
OUTPUT_DIR = ROOT / "dataset_candidates" / "batch9_standing"
OUTPUT_CSV = ROOT / "metadata" / "batch9_standing_downloaded.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 30
TIMEOUT = 20

with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
    candidates = list(csv.DictReader(f))

downloaded = []
failed = []

for row in candidates:
    if len(downloaded) >= TARGET:
        break

    name = row["file_name"]
    path = OUTPUT_DIR / name

    try:
        if path.exists():
            with Image.open(path) as img:
                img.verify()
        else:
            url = row["coco_url"]

            if not url:
                failed.append(name)
                continue

            response = requests.get(url, timeout=TIMEOUT)
            response.raise_for_status()

            with Image.open(BytesIO(response.content)) as img:
                img.verify()

            path.write_bytes(response.content)

        downloaded.append({
            "file_name": name,
            "image_id": row["image_id"],
            "source": "MS COCO 2017",
            "status": "downloaded"
        })

        print(f"[{len(downloaded)}/{TARGET}] OK: {name}")

    except Exception as e:
        failed.append(name)
        print(f"FAILED: {name} - {e}")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["file_name", "image_id", "source", "status"]
    )
    writer.writeheader()
    writer.writerows(downloaded)

print("\n===== BATCH 9 DOWNLOAD REPORT =====")
print("Downloaded:", len(downloaded))
print("Failed:", len(failed))
print("Saved to:", OUTPUT_DIR)
