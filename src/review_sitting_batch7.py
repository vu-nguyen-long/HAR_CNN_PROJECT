
from pathlib import Path
import csv
import cv2

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "dataset_candidates" / "batch7_sitting"
OUTPUT = ROOT / "metadata" / "batch7_sitting_review.csv"

# Doc ket qua da kiem tra neu co
reviews = {}

if OUTPUT.exists():
    with open(OUTPUT, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviews[row["file_name"]] = row["review"]

images = sorted(IMAGE_DIR.glob("*.jpg"))

print("1: Sitting hop le")
print("2: Khong phai Sitting / khong ro")
print("Q: Luu va thoat")

for path in images:
    if path.name in reviews:
        continue

    img = cv2.imread(str(path))
    if img is None:
        print("Khong doc duoc:", path.name)
        continue

    h, w = img.shape[:2]
    scale = min(900 / w, 650 / h, 1.0)
    img = cv2.resize(img, None, fx=scale, fy=scale)

    cv2.imshow("Review Sitting Batch 7", img)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("1"), ord("2")):
            reviews[path.name] = (
                "approved" if key == ord("1") else "rejected"
            )

            with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file_name", "review"])
                for name, status in sorted(reviews.items()):
                    writer.writerow([name, status])

            print(path.name, "->", reviews[path.name])
            break

        if key in (ord("q"), ord("Q"), 27):
            cv2.destroyAllWindows()
            print("Da luu tien do.")
            raise SystemExit

cv2.destroyAllWindows()

approved = sum(v == "approved" for v in reviews.values())
print("Hoan thanh!")
print("Sitting hop le:", approved)
print("Anh bi loai:", sum(v == "rejected" for v in reviews.values()))
