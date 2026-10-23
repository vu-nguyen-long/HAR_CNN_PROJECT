
import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# 1. Xác định thư mục dự án
PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_DIR / "metadata" / "sitting_keypoints_batch6.csv"
OUTPUT_DIR = PROJECT_DIR / "dataset_candidates"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Đọc danh sách ảnh chứa person
df = pd.read_csv(CSV_PATH, dtype={"file_name": str})

print("Tổng số ảnh có người:", len(df))



# 3. Chon 100 anh uu tien tu danh sach da loc

LABEL_PATH = PROJECT_DIR / "metadata" / "har_labels.csv"

# Anh da tai trong dataset_candidates
existing_files = {
    p.name for p in OUTPUT_DIR.glob("*.jpg")
}

# Anh da chuyen vao dataset_raw
RAW_DIR = PROJECT_DIR / "dataset_raw"

raw_files = {
    p.name for p in RAW_DIR.rglob("*.jpg")
}

# Anh da gan nhan
labeled_files = set()

if LABEL_PATH.exists():
    labels_df = pd.read_csv(
        LABEL_PATH,
        dtype={"file_name": str}
    )
    labeled_files = set(
        labels_df["file_name"].dropna()
    )

# Loai bo tat ca anh da xu ly
excluded_files = existing_files | raw_files | labeled_files

labeled_files = set()

if LABEL_PATH.exists():
    labels_df = pd.read_csv(
        LABEL_PATH,
        dtype={"file_name": str}
    )
    labeled_files = set(labels_df["file_name"])

excluded_files = existing_files | labeled_files

available_df = df[
    ~df["file_name"].isin(excluded_files)
].copy()

# Giu thu tu uu tien tu file CSV
sample_df = available_df.copy()

print("Tong anh ung vien:", len(df))
print("Anh con lai:", len(available_df))
print("So anh se tai:", len(sample_df))



# 4. Tạo danh sách kết quả tải
download_results = []

# 5. Tải từng ảnh
for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):

    file_name = row["file_name"]

    image_url = (
    "https://s3.amazonaws.com/images.cocodataset.org/val2017/"
    + file_name
)
    

    save_path = OUTPUT_DIR / file_name

    status = "success"

    try:
        if not save_path.exists():
            response = requests.get(
                image_url,
                timeout=30
            )
            response.raise_for_status()
            save_path.write_bytes(response.content)

        # Kiểm tra ảnh có mở được không
        with Image.open(save_path) as img:
            img.verify()

    except Exception as e:
        status = "failed"
        print(f"\nLỗi tải {file_name}: {e}")

        if save_path.exists():
            save_path.unlink()

    download_results.append({
        "image_id": row["image_id"],
        "file_name": file_name,
        "download_status": status
    })

# 6. Lưu kết quả tải
result_df = pd.DataFrame(download_results)

result_path = (
    PROJECT_DIR / "metadata" / "download_results_keypoints_batch6.csv"
)

result_df.to_csv(result_path, index=False)

# 7. Thống kê
print("\nKẾT QUẢ TẢI ẢNH")
print(result_df["download_status"].value_counts())
print("Thư mục lưu ảnh:", OUTPUT_DIR)
