
from pathlib import Path
from collections import Counter
from PIL import Image, ImageOps
import csv
import statistics

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset_final"
OUTPUT = ROOT / "metadata" / "image_statistics.csv"

CLASSES = ["standing", "sitting", "moving"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

records = []
errors = []

for label in CLASSES:
    folder = DATASET / label

    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue

        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                width, height = img.size
                img.load()

                if width <= 0 or height <= 0:
                    raise ValueError("Invalid image dimensions")

                ratio = width / height

                if ratio > 1.05:
                    orientation = "landscape"
                elif ratio < 0.95:
                    orientation = "portrait"
                else:
                    orientation = "square"

                records.append({
                    "file_name": path.name,
                    "label": label,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(ratio, 4),
                    "orientation": orientation
                })

        except Exception as e:
            errors.append((str(path), str(e)))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name", "label", "width", "height",
            "aspect_ratio", "orientation"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

print("\n===== IMAGE STATISTICS REPORT =====")

counts = Counter(r["label"] for r in records)

for label in CLASSES:
    print(f"{label}: {counts[label]} images")

print("Total valid images:", len(records))
print("Image errors:", len(errors))

if records:
    widths = [r["width"] for r in records]
    heights = [r["height"] for r in records]
    ratios = [r["aspect_ratio"] for r in records]

    print("\nWidth min/max:", min(widths), max(widths))
    print("Height min/max:", min(heights), max(heights))
    print("Average width:", round(statistics.mean(widths), 2))
    print("Average height:", round(statistics.mean(heights), 2))
    print("Aspect ratio min/max:", min(ratios), max(ratios))
    print("Average aspect ratio:", round(statistics.mean(ratios), 4))

    print("\nOrientation distribution:")
    orientations = Counter(r["orientation"] for r in records)

    for orientation in ["landscape", "portrait", "square"]:
        print(f"{orientation}: {orientations[orientation]}")

print("\nCSV saved:", OUTPUT)

if errors:
    print("\nImage errors:")
    for path, error in errors[:10]:
        print(path, error)

if len(records) == 450 and not errors and all(
    counts[label] == 150 for label in CLASSES
):
    print("\nSTATUS: SUCCESS")
else:
    print("\nSTATUS: CHECK REQUIRED")
