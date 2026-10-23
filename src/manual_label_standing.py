
from pathlib import Path
from PIL import Image, ImageOps
import csv

ROOT = Path(__file__).resolve().parent.parent

LABELS_CSV = ROOT / "metadata" / "har_labels.csv"
REVIEW_CSV = ROOT / "metadata" / "standing_review.csv"

SEARCH_DIRS = [
    ROOT / "dataset_candidates",
    ROOT / "dataset_raw",
]

TARGET = 7

with open(LABELS_CSV, newline="", encoding="utf-8-sig") as f:
    uncertain = [
        row for row in csv.DictReader(f)
        if row["label"].strip().lower() == "uncertain"
    ]

# Tim duong dan anh
image_index = {}

for folder in SEARCH_DIRS:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }:
            image_index.setdefault(path.name, path)

# Doc tien do da luu
reviews = {}

if REVIEW_CSV.exists():
    with open(REVIEW_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            reviews[row["file_name"]] = row["label"]

def save():
    with open(REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "label"])
        for name, label in reviews.items():
            writer.writerow([name, label])

def standing_count():
    return sum(label == "standing" for label in reviews.values())

if standing_count() >= TARGET:
    print("Already reached target:", standing_count())
    raise SystemExit

for index, row in enumerate(uncertain, 1):
    name = row["file_name"]

    if name in reviews:
        continue

    path = image_index.get(name)

    if path is None:
        print("Missing:", name)
        continue

    try:
        with Image.open(path) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
            img.thumbnail((900, 650))

        print(f"\nImage {index}/{len(uncertain)}: {name}")
        print(f"Standing accepted: {standing_count()}/{TARGET}")

        img.show()

        print("1=Standing | 2=Not suitable | Q=Save and quit")

        while True:
            choice = input("Your label: ").strip().lower()

            if choice == "q":
                save()
                print("Progress saved:", REVIEW_CSV)
                raise SystemExit

            if choice == "1":
                reviews[name] = "standing"
                save()
                break

            if choice == "2":
                reviews[name] = "uncertain"
                save()
                break

            print("Invalid choice.")

    except (OSError, ValueError) as e:
        print("Image error:", name, e)

    if standing_count() >= TARGET:
        print("\nTarget reached: 7 standing images!")
        break

save()

print("\n===== STANDING REVIEW SUMMARY =====")
print("Standing:", standing_count())
print("Reviewed:", len(reviews))
print("Saved to:", REVIEW_CSV)
