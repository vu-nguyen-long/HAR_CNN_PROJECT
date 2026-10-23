
from pathlib import Path
from collections import defaultdict
import json
import pandas as pd

# 1. Duong dan
PROJECT_DIR = Path(__file__).resolve().parent.parent

ANNOTATION_PATH = (
    PROJECT_DIR / "annotations" / "instances_val2017.json"
)

IMAGE_DIR = PROJECT_DIR / "dataset_candidates"
LABEL_PATH = PROJECT_DIR / "metadata" / "har_labels.csv"

OUTPUT_PATH = (
    PROJECT_DIR / "metadata" / "har_candidates_filtered.csv"
)

# 2. Doc annotations
with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

person_ids = {
    cat["id"]
    for cat in coco["categories"]
    if cat["name"] == "person"
}

# Gom bounding box theo image_id
person_boxes = defaultdict(list)

for ann in coco["annotations"]:
    if ann["category_id"] in person_ids:
        person_boxes[ann["image_id"]].append(ann["bbox"])

# 3. Lay danh sach anh da xu ly
excluded_files = {
    p.name for p in IMAGE_DIR.glob("*.jpg")
}

if LABEL_PATH.exists():
    labels_df = pd.read_csv(
        LABEL_PATH,
        dtype={"file_name": str}
    )
    excluded_files.update(labels_df["file_name"].dropna())

# 4. Loc anh ung vien
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
    image_area = width * height

    if image_area <= 0:
        continue

    # Uu tien anh co 1 hoac 2 nguoi
    if not (1 <= len(boxes) <= 2):
        continue

    # Lay bounding box nguoi lon nhat
    largest_box = max(
        boxes,
        key=lambda b: b[2] * b[3]
    )

    x, y, box_w, box_h = largest_box

    area_ratio = (box_w * box_h) / image_area
    height_ratio = box_h / height

    # Dieu kien loc
    if area_ratio < 0.10:
        continue

    if height_ratio < 0.35:
        continue

    candidates.append({
        "image_id": image_id,
        "file_name": file_name,
        "person_count": len(boxes),
        "area_ratio": round(area_ratio, 4),
        "height_ratio": round(height_ratio, 4)
    })

# 5. Sap xep uu tien
result_df = pd.DataFrame(
    candidates,
    columns=[
        "image_id",
        "file_name",
        "person_count",
        "area_ratio",
        "height_ratio"
    ]
)

if not result_df.empty:
    result_df = result_df.sort_values(
        by=["person_count", "area_ratio"],
        ascending=[True, False]
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False)

# 6. Thong ke
print("\n=== KET QUA LOC ANH HAR ===")
print("Tong anh COCO:", len(images))
print("Anh da loai tru:", len(excluded_files))
print("Anh ung vien phu hop:", len(result_df))
print("File ket qua:", OUTPUT_PATH)

if not result_df.empty:
    print("\n10 anh uu tien dau tien:")
    print(result_df.head(10).to_string(index=False))
else:
    print("Khong tim thay anh phu hop voi nguong hien tai.")
