
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
import math

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "dataset_candidates" / "batch9_standing"
SELECTED = ROOT / "dataset_final" / "standing"
OUTPUT = ROOT / "metadata" / "standing_replacement_candidates.jpg"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

selected_names = {
    p.name for p in SELECTED.iterdir()
    if p.is_file()
}

candidates = sorted(
    p for p in SOURCE.iterdir()
    if p.is_file()
    and p.suffix.lower() in EXTENSIONS
    and p.name not in selected_names
)

print("Available candidates:", len(candidates))

if not candidates:
    print("No unused candidates found.")
    raise SystemExit(0)

cols = 4
cell_w = 240
cell_h = 220
rows = math.ceil(len(candidates) / cols)

sheet = Image.new(
    "RGB",
    (cols * cell_w, rows * cell_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, path in enumerate(candidates):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((220, 180))

        x = (i % cols) * cell_w
        y = (i // cols) * cell_h

        px = x + (cell_w - img.width) // 2
        py = y + 5

        sheet.paste(img, (px, py))
        draw.text(
            (x + 10, y + 190),
            path.name,
            fill="black"
        )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT, quality=95)

print("Contact sheet saved:", OUTPUT)
