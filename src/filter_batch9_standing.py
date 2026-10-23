
from pathlib import Path
import json
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent

ANNOTATION = ROOT / "annotations" / "person_keypoints_val2017.json"
LABELS_CSV = ROOT / "metadata" / "har_labels.csv"
OUTPUT_CSV = ROOT / "metadata" / "batch9_standing_candidates.csv"

with open(ANNOTATION, encoding="utf-8") as f:
    coco = json.load(f)

with open(LABELS_CSV, newline="", encoding="utf-8-sig") as f:
    used_names = {
        row["file_name"]
        for row in csv.DictReader(f)
    }

images = {
    img["id"]: img
    for img in coco["images"]
}

annotations = defaultdict(list)

for ann in coco["annotations"]:
    annotations[ann["image_id"]].append(ann)

candidates = []

for image_id, people in annotations.items():
    image = images[image_id]
    filename = image["file_name"]

    if filename in used_names:
        continue

    # Uu tien anh co mot nguoi
    if len(people) != 1:
        continue

    person = people[0]
    kpts = person["keypoints"]

    def point(index):
        x, y, visibility = kpts[index * 3:index * 3 + 3]
        return (x, y) if visibility > 0 else None

    left_shoulder = point(5)
    right_shoulder = point(6)
    left_hip = point(11)
    right_hip = point(12)
    left_ankle = point(15)
    right_ankle = point(16)

    required = [
        left_shoulder, right_shoulder,
        left_hip, right_hip,
        left_ankle, right_ankle
    ]

    if any(p is None for p in required):
        continue

    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    ankle_y = (left_ankle[1] + right_ankle[1]) / 2

    torso = hip_y - shoulder_y
    legs = ankle_y - hip_y

    if torso <= 0 or legs <= 0:
        continue

    # Heuristic: chan duoi hong va du dai
    if legs / torso < 1.2:
        continue

    candidates.append({
        "image_id": image_id,
        "file_name": filename,
        "coco_url": image.get("coco_url", ""),
        "leg_torso_ratio": round(legs / torso, 2)
    })

candidates.sort(
    key=lambda row: row["leg_torso_ratio"],
    reverse=True
)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_id", "file_name",
            "coco_url", "leg_torso_ratio"
        ]
    )
    writer.writeheader()
    writer.writerows(candidates)

print("\n===== STANDING CANDIDATE REPORT =====")
print("Candidates found:", len(candidates))
print("Saved:", OUTPUT_CSV)
print("Top 5 candidates:")

for row in candidates[:5]:
    print(row["file_name"], row["leg_torso_ratio"])
