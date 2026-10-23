
from pathlib import Path
import csv
import cv2

# 1. Xac dinh duong dan
PROJECT_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_DIR / "dataset_candidates"
LABEL_FILE = PROJECT_DIR / "metadata" / "har_labels.csv"

# 2. Khai bao nhan
LABELS = {
    ord("1"): "standing",
    ord("2"): "sitting",
    ord("3"): "moving",
    ord("4"): "uncertain"
}

LABEL_FILE.parent.mkdir(parents=True, exist_ok=True)

# 3. Doc nhan da luu
saved_labels = {}

if LABEL_FILE.exists():
    with open(LABEL_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            saved_labels[row["file_name"]] = row["label"]

# 4. Lay danh sach anh chua gan nhan
images = sorted(IMAGE_DIR.glob("*.jpg"))

images = [
    image for image in images
    if image.name != "test_coco.jpg"
    and image.name not in saved_labels
]

print("So anh chua gan nhan:", len(images))
print("1: Standing")
print("2: Sitting")
print("3: Moving")
print("4: Uncertain")
print("Q: Luu va thoat")

# 5. Hien thi anh va gan nhan
for image_path in images:

    img = cv2.imread(str(image_path))

    if img is None:
        print("Khong doc duoc anh:", image_path.name)
        continue

    height, width = img.shape[:2]

    scale = min(
        900 / width,
        650 / height,
        1.0
    )

    img = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale
    )

    cv2.imshow("HAR Data Labeling", img)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in LABELS:
            label = LABELS[key]
            saved_labels[image_path.name] = label

            # Luu ket qua sau moi anh
            with open(
                LABEL_FILE,
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)
                writer.writerow(["file_name", "label"])

                for name, value in sorted(saved_labels.items()):
                    writer.writerow([name, value])

            print(image_path.name, "->", label)
            break

        elif key in (ord("q"), ord("Q"), 27):
            cv2.destroyAllWindows()
            print("Da luu tien do. Thoat.")
            raise SystemExit

cv2.destroyAllWindows()
print("Hoan thanh gan nhan!")
