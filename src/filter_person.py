
import json
import pandas as pd
from pathlib import Path

# 1. Xác định đường dẫn thư mục dự án
project_dir = Path(__file__).resolve().parent.parent

annotation_path = (
    project_dir / "annotations" / "instances_val2017.json"
)

output_path = (
    project_dir / "metadata" / "person_images.csv"
)

# 2. Đọc file annotation JSON
with open(annotation_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

print("Đọc annotation thành công!")

# 3. Tìm category ID của đối tượng person
person_id = None

for category in coco["categories"]:
    if category["name"] == "person":
        person_id = category["id"]
        break

if person_id is None:
    raise ValueError("Không tìm thấy category person")

print("Category ID của person:", person_id)

# 4. Tìm tất cả image_id chứa person
person_image_ids = set()

for annotation in coco["annotations"]:
    if annotation["category_id"] == person_id:
        person_image_ids.add(annotation["image_id"])

print("Số ảnh chứa person:", len(person_image_ids))

# 5. Lấy thông tin các ảnh có người
person_images = []

for image in coco["images"]:
    if image["id"] in person_image_ids:
        person_images.append({
            "image_id": image["id"],
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
            "source_split": "val2017"
        })

# 6. Chuyển dữ liệu sang DataFrame
df = pd.DataFrame(person_images)

# Sắp xếp theo image_id
df = df.sort_values(by="image_id")

# 7. Lưu thành file CSV
output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

# 8. Hiển thị kết quả
print("\n5 ảnh đầu tiên:")
print(df.head())

print("\nTổng số ảnh được lọc:", len(df))
print("Đã lưu file:", output_path)
