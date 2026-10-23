
from pathlib import Path
from PIL import Image, ImageOps
import csv

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset_candidates" / "batch9_standing"
INPUT_CSV = ROOT / "metadata" / "batch9_standing_downloaded.csv"
OUTPUT_CSV = ROOT / "metadata" / "batch9_standing_review.csv"

TARGET = 7

KEYS = {
    "1": "standing",
    "2": "sitting",
    "3": "moving",
    "4": "uncertain"
}

# Doc danh sach anh da tai
with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
    images = list(csv.DictReader(f))

# Khoi phuc tien do neu da gan nhan truoc do
reviews = {}

if OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            reviews[row["file_name"]] = row["label"]

def save_progress():
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "label"])
        for name, label in reviews.items():
            writer.writerow([name, label])

def standing_count():
    return sum(label == "standing" for label in reviews.values())

print("\n===== BATCH 9 MANUAL LABELING =====")
print("Total candidate images:", len(images))
print("Previously reviewed:", len(reviews))
print("Standing accepted:", standing_count())

if standing_count() >= TARGET:
    print("Target already reached!")
    raise SystemExit

for index, row in enumerate(images, 1):
    filename = row["file_name"]

    if filename in reviews:
        continue

    path = IMAGE_DIR / filename

    if not path.exists():
        print("Missing:", filename)
        continue

    try:
        with Image.open(path) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
            img.thumbnail((900, 650))

        print(f"\nImage {index}/{len(images)}: {filename}")
        print(f"Standing accepted: {standing_count()}/{TARGET}")

        img.show()

        print("1=Standing | 2=Sitting | 3=Moving")
        print("4=Uncertain | Q=Save and quit")

        while True:
            choice = input("Your label: ").strip().lower()

            if choice == "q":
                save_progress()
                print("Progress saved:", OUTPUT_CSV)
                raise SystemExit

            if choice in KEYS:
                reviews[filename] = KEYS[choice]
                save_progress()
                break

            print("Invalid choice. Try again.")

    except (OSError, ValueError) as e:
        print("Image error:", filename, e)

    if standing_count() >= TARGET:
        print("\nTarget reached: 7 standing images!")
        break

save_progress()

print("\n===== BATCH 9 REVIEW SUMMARY =====")

for label in KEYS.values():
    count = sum(v == label for v in reviews.values())
    print(f"{label}: {count}")

print("Total reviewed:", len(reviews))
print("Saved to:", OUTPUT_CSV)
