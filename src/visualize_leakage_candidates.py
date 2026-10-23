
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import csv
import math

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset"
REPORT = ROOT / "metadata" / "resnet_similarity_report.csv"
OUTPUT_DIR = ROOT / "metadata" / "leakage_review"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_W = 300
IMAGE_H = 220
ROW_H = 270
PAIRS_PER_PAGE = 10

with open(REPORT, newline="", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))

def make_thumbnail(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((CELL_W - 20, IMAGE_H))
        return img.copy()

for page_start in range(0, len(pairs), PAIRS_PER_PAGE):
    page_pairs = pairs[
        page_start:page_start + PAIRS_PER_PAGE
    ]

    sheet = Image.new(
        "RGB",
        (CELL_W * 2, ROW_H * len(page_pairs)),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for row_index, pair in enumerate(page_pairs):
        pair_index = page_start + row_index + 1
        y = row_index * ROW_H

        path_a = (
            DATASET
            / pair["split_a"]
            / pair["label_a"]
            / pair["file_a"]
        )

        path_b = (
            DATASET
            / pair["split_b"]
            / pair["label_b"]
            / pair["file_b"]
        )

        for col, path in enumerate((path_a, path_b)):
            image = make_thumbnail(path)

            x = col * CELL_W + (CELL_W - image.width) // 2
            py = y + 5

            sheet.paste(image, (x, py))

        draw.text(
            (10, y + 225),
            f"#{pair_index} | {pair['file_a']} "
            f"({pair['split_a']})",
            fill="black"
        )

        draw.text(
            (CELL_W + 10, y + 225),
            f"{pair['file_b']} ({pair['split_b']})",
            fill="black"
        )

        draw.text(
            (10, y + 245),
            f"Cosine similarity: "
            f"{float(pair['similarity']):.4f}",
            fill="black"
        )

    page_number = page_start // PAIRS_PER_PAGE + 1

    output = OUTPUT_DIR / f"leakage_review_{page_number}.jpg"
    sheet.save(output, quality=95)

    print("Saved:", output)

print("\nTotal pairs:", len(pairs))
print("STATUS: COMPLETED")
