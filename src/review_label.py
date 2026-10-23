
from pathlib import Path
import csv
import shutil
import cv2

# 1. Duong dan du an
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

VALID_LABELS = {"standing", "sitting", "moving"}

# 3. Kiem tra file CSV
if not LABEL_FILE.exists():
    raise FileNotFoundError("Khong tim thay har_labels.csv")

# Sao luu CSV truoc khi chinh sua
BACKUP_FILE = LABEL_FILE.with_name("har_labels_backup.csv")

if not BACKUP_FILE.exists():
    shutil.copy2(LABEL_FILE, BACKUP_FILE)
    print("Da sao luu:", BACKUP_FILE)

# 4. Doc du lieu nhan
with open(LABEL_FILE, "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if not {"file_name", "label"}.issubset(reader.fieldnames or []):
    raise ValueError("CSV thieu cot file_name hoac label")

# Chi xem lai 109 anh thuoc 3 class
review_indices = [
    i for i, row in enumerate(rows)
    if row["label"] in VALID_LABELS
]

print("So anh can kiem tra:", len(review_indices))
print("1: Standing | 2: Sitting | 3: Moving")
print("4: Uncertain | Enter/Space: Giu nhan")
print("Q: Luu va thoat")

# 5. Ham luu nhan
def save_labels():
    temp_file = LABEL_FILE.with_suffix(".tmp")

    with open(temp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=reader.fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)

    temp_file.replace(LABEL_FILE)

# 6. Hien thi va kiem tra tung anh
for number, index in enumerate(review_indices, start=1):
    row = rows[index]
    image_path = IMAGE_DIR / row["file_name"]

    img = cv2.imread(str(image_path))

    if img is None:
        print("Khong doc duoc:", image_path.name)
        continue

    height, width = img.shape[:2]
    scale = min(900 / width, 650 / height, 1.0)

    img = cv2.resize(
        img, None, fx=scale, fy=scale
    )

    # Tao thanh thong tin phia tren anh
    img = cv2.copyMakeBorder(
        img, 75, 0, 0, 0,
        cv2.BORDER_CONSTANT,
        value=(35, 35, 35)
    )

    current_label = row["label"]

    cv2.putText(
        img,
        f"{number}/{len(review_indices)} | Label: {current_label}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        img,
        "1:Stand 2:Sit 3:Move 4:Uncertain Enter:Keep Q:Quit",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 255, 200),
        1
    )

    cv2.imshow("HAR Label Review", img)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in LABELS:
            new_label = LABELS[key]
            rows[index]["label"] = new_label
            save_labels()

            print(
                image_path.name,
                current_label,
                "->",
                new_label
            )
            break

        elif key in (13, 10, 32):
            print(image_path.name, "-> Giu nguyen")
            break

        elif key in (ord("q"), ord("Q"), 27):
            cv2.destroyAllWindows()
            print("Da thoat. Moi thay doi da duoc luu.")
            raise SystemExit

cv2.destroyAllWindows()
print("Hoan thanh kiem tra nhan!")
