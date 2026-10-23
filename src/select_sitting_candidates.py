
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

ANNOTATION_PATH = PROJECT_DIR / "annotations" / "instances_val2017.json"
LABEL_PATH = PROJECT_DIR / "metadata" / "har_labels.csv"
IMAGE_DIR = PROJECT_DIR / "dataset_candidates"
RAW_DIR = PROJECT_DIR / "dataset_raw"

OUTPUT_PATH = PROJECT_DIR / "metadata" / "sitting_candidates_batch5.csv"

# 1. Doc annotations
with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

person_ids = {
    cat["id"]
    for cat in coco["categories"]
    if cat["name"] == "person"
}

person_boxes = defaultdict(list)

for ann in coco["annotations"]:
    if ann["category_id"] in person_ids:
        person_boxes[ann["image_id"]].append(ann["bbox"])

# 2. Loai anh da xu ly
excluded_files = {
    p.name for p in IMAGE_DIR.glob("*.jpg")
}

excluded_files.update(
    p.name for p in RAW_DIR.rglob("*.jpg")
)

if LABEL_PATH.exists():
    labels_df = pd.read_csv(
        LABEL_PATH,
        dtype={"file_name": str}
    )
    excluded_files.update(
        labels_df["file_name"].dropna()
    )

# 3. Loc anh ung vien
candidates = []

for image_id, boxes in person_boxes.items():

    img = images.get(image_id)

    if img is None:
        continue

    file_name = img["file_name"]

    if file_name in excluded_files:
        continue

    width = img["width"]
    height = img["height"]

    if width <= 0 or height <= 0:
        continue

    if not (1 <= len(boxes) <= 3):
        continue

    largest_box = max(
        boxes,
        key=lambda b: b[2] * b[3]
    )

    x, y, box_w, box_h = largest_box

    if box_w <= 0 or box_h <= 0:
        continue

    area_ratio = (box_w * box_h) / (width * height)
    height_ratio = box_h / height
    aspect_ratio = box_w / box_h

    if area_ratio < 0.05:
        continue

    if height_ratio < 0.20:
        continue

    # Diem uu tien tu the ngoi (heuristic)
    sitting_score = 0

    if aspect_ratio >= 0.55:
        sitting_score += 2

    if aspect_ratio >= 0.75:
        sitting_score += 2

    if 0.20 <= height_ratio <= 0.65:
        sitting_score += 1

    candidates.append({
        "image_id": image_id,
        "file_name": file_name,
        "person_count": len(boxes),
        "area_ratio": round(area_ratio, 4),
        "height_ratio": round(height_ratio, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "sitting_score": sitting_score
    })

# 4. Sap xep uu tien
result_df = pd.DataFrame(candidates)

if not result_df.empty:
    result_df = result_df.sort_values(
        by=["sitting_score", "area_ratio"],
        ascending=[False, False]
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False)

print("\n=== SITTING CANDIDATES BATCH 5 ===")
print("Tong anh COCO:", len(images))
print("Anh da xu ly:", len(excluded_files))
print("Anh ung vien moi:", len(result_df))
print("Da luu:", OUTPUT_PATH)

if not result_df.empty:
    print(result_df.head(10).to_string(index=False))
