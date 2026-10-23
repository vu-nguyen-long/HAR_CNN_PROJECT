
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import csv

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset_candidates" / "batch8_sitting"
INPUT_CSV = ROOT / "metadata" / "batch8_sitting.csv"
OUTPUT_CSV = ROOT / "metadata" / "batch8_sitting_review.csv"

KEYS = {
    "1": "sitting",
    "2": "standing",
    "3": "moving",
    "4": "uncertain"
}

TARGET = 27

with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
    images = list(csv.DictReader(f))

# Khoi phuc tien do cu
reviews = {}

if OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            reviews[row["file_name"]] = row["label"]

def save():
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "label"])
        for name, label in reviews.items():
            writer.writerow([name, label])

def sitting_count():
    return sum(v == "sitting" for v in reviews.values())

for index, row in enumerate(images, 1):
    name = row["file_name"]

    if name in reviews:
        continue

    path = IMAGE_DIR / name

    if not path.exists():
        print("Missing:", name)
        continue

    try:
        with Image.open(path) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
            img.thumbnail((900, 650))

        print(f"\nImage {index}/{len(images)}: {name}")
        print(f"Sitting accepted: {sitting_count()}/{TARGET}")

        img.show()

        print("1=Sitting | 2=Standing | 3=Moving")
        print("4=Uncertain | Q=Save and quit")

        while True:
            choice = input("Your label: ").strip().lower()

            if choice == "q":
                save()
                print("Progress saved:", OUTPUT_CSV)
                raise SystemExit

            if choice in KEYS:
                reviews[name] = KEYS[choice]
                save()
                break

            print("Invalid choice. Try again.")

    except (OSError, ValueError) as e:
        print("Image error:", name, e)

    if sitting_count() >= TARGET:
        print("\nTarget reached: 27 sitting images!")
        break

save()

print("\n===== REVIEW SUMMARY =====")
for label in KEYS.values():
    count = sum(v == label for v in reviews.values())
    print(f"{label}: {count}")

print("Saved to:", OUTPUT_CSV)
