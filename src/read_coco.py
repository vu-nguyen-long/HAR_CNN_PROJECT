
import json
from pathlib import Path

# Đường dẫn đến file annotation
annotation_path = Path("annotations/instances_val2017.json")

# Kiểm tra file có tồn tại không
if not annotation_path.exists():
    print("Không tìm thấy file annotation!")
    raise SystemExit(1)

# Đọc dữ liệu JSON
with open(annotation_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

# Hiển thị thông tin tổng quan
print("Đọc dữ liệu MS-COCO thành công!")

print("Tổng số ảnh:", len(coco["images"]))
print("Tổng số annotation:", len(coco["annotations"]))
print("Tổng số category:", len(coco["categories"]))

# Xem thông tin ảnh đầu tiên
print("\nThông tin ảnh đầu tiên:")
print(coco["images"][0])
