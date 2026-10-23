
from pathlib import Path
from PIL import Image, ImageDraw
import csv
import math

ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = ROOT / "dataset_candidates" / "batch8_sitting"
CSV_PATH = ROOT / "metadata" / "batch8_sitting.csv"
OUTPUT_DIR = ROOT / "annotations"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

COLS = 5
ROWS = 4
PAGE_SIZE = COLS * ROWS

CELL_W = 200
CELL_H = 190
IMG_H = 155

pages = math.ceil(len(rows) / PAGE_SIZE)

for page in range(pages):
    subset = rows[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]

    sheet = Image.new(
        "RGB",
        (COLS * CELL_W, ROWS * CELL_H),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for i, row in enumerate(subset):
        path = IMAGE_DIR / row["file_name"]

        col = i % COLS
        r = i // COLS
        x = col * CELL_W
        y = r * CELL_H

        if path.exists():
            try:
                with Image.open(path) as source:
                    img = source.convert("RGB")
                    img.thumbnail((CELL_W - 10, IMG_H))
                    sheet.paste(img, (x + 5, y + 5))
            except Exception:
                draw.text((x + 5, y + 5), "IMAGE ERROR", fill="red")
        else:
            draw.text((x + 5, y + 5), "MISSING", fill="red")

        draw.text(
            (x + 5, y + IMG_H + 8),
            row["file_name"],
            fill="black"
        )

    output = OUTPUT_DIR / f"batch8_sheet_{page + 1}.jpg"
    sheet.save(output, quality=95)
    print("Created:", output)

print("Total pages:", pages)
