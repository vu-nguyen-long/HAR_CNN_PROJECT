
from pathlib import Path
import json
import math
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

ANNOTATION_PATH = (
    PROJECT_DIR / "annotations" / "person_keypoints_val2017.json"
)

LABEL_PATH = PROJECT_DIR / "metadata" / "har_labels.csv"
IMAGE_DIR = PROJECT_DIR / "dataset_candidates"
RAW_DIR = PROJECT_DIR / "dataset_raw"

OUTPUT_PATH = (
    PROJECT_DIR / "metadata" / "sitting_keypoints_batch6.csv"
)

# Tinh goc giua 3 diem A-B-C
def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot = ba[0] * bc[0] + ba[1] * bc[1]
    norm_ba = math.hypot(*ba)
    norm_bc = math.hypot(*bc)

    if norm_ba == 0 or norm_bc == 0:
        return None

    cos_angle = dot / (norm_ba * norm_bc)
    cos_angle = max(-1, min(1, cos_angle))

    return math.degrees(math.acos(cos_angle))


# Doc annotation COCO
with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
    coco = json.load(f)

images = {img["id"]: img for img in coco["images"]}

# Anh da xu ly
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

candidates = []

# COCO keypoint indices:
# 11 left hip, 12 right hip
# 13 left knee, 14 right knee
# 15 left ankle, 16 right ankle

for ann in coco["annotations"]:

    if ann.get("iscrowd", 0):
        continue

    if ann.get("num_keypoints", 0) < 6:
        continue

    image_id = ann["image_id"]
    img = images.get(image_id)

    if img is None:
        continue

    file_name = img["file_name"]

    if file_name in excluded_files:
        continue

    keypoints = ann["keypoints"]

    def get_point(index):
        x, y, v = keypoints[index * 3:index * 3 + 3]
        if v < 2:
            return None
        return (x, y)

    angles = []

    for hip, knee, ankle in [
        (11, 13, 15),
        (12, 14, 16)
    ]:
        a = get_point(hip)
        b = get_point(knee)
        c = get_point(ankle)

        if a and b and c:
            angle = calculate_angle(a, b, c)

            if angle is not None:
                angles.append(angle)

    if not angles:
        continue

    # Uu tien chan gap tu 60 den 130 do
    bent_angles = [
        angle for angle in angles
        if 60 <= angle <= 130
    ]

    if not bent_angles:
        continue

    best_angle = min(
        bent_angles,
        key=lambda x: abs(x - 90)
    )

    candidates.append({
        "image_id": image_id,
        "file_name": file_name,
        "knee_angle": round(best_angle, 2),
        "sitting_score": round(
            1 - abs(best_angle - 90) / 90, 4
        )
    })

# Giu moi anh mot lan
result_df = pd.DataFrame(
    candidates,
    columns=[
        "image_id",
        "file_name",
        "knee_angle",
        "sitting_score"
    ]
)

if not result_df.empty:
    result_df = result_df.sort_values(
        "sitting_score",
        ascending=False
    ).drop_duplicates("file_name")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(OUTPUT_PATH, index=False)

print("\n=== KET QUA LOC SITTING BANG KEYPOINTS ===")
print("Tong anh COCO:", len(images))
print("Anh da xu ly:", len(excluded_files))
print("Anh ung vien moi:", len(result_df))
print("File ket qua:", OUTPUT_PATH)

if not result_df.empty:
    print("\n10 anh uu tien:")
    print(result_df.head(10).to_string(index=False))
