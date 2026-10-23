# HAR CNN — Dataset Documentation

## 1. Project Overview

Dự án xây dựng bộ dữ liệu ảnh phục vụ huấn luyện mô hình CNN nhận dạng hoạt động của con người (Human Activity Recognition).

Bộ dữ liệu gồm 3 lớp:
- `standing`: Đứng.
- `sitting`: Ngồi.
- `moving`: Di chuyển, bao gồm các hoạt động như đi bộ và chạy.

## 2. Dataset Information

- Tổng số ảnh: **450**
- Số lớp: **3**
- Số ảnh mỗi lớp: **150**
- Tỷ lệ chia dữ liệu: **80% Train / 10% Validation / 10% Test**
- Random seed: **42**

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Standing | 120 | 15 | 15 | 150 |
| Sitting | 120 | 15 | 15 | 150 |
| Moving | 120 | 15 | 15 | 150 |
| **Total** | **360** | **45** | **45** | **450** |

## 3. Data Preprocessing

Pipeline preprocessing được triển khai tại `src/preprocessing.py`.

Các bước:
1. Điều chỉnh hướng ảnh theo EXIF.
2. Chuyển ảnh sang RGB.
3. Resize giữ nguyên tỷ lệ khung hình.
4. Padding để đạt kích thước `224 × 224`.
5. Chuyển dữ liệu thành NumPy array kiểu `float32`.
6. Chuẩn hóa pixel về `[0, 1]` khi `normalize=True`.

Đầu ra: `(224, 224, 3)`.

## 4. Dataset Structure

Dữ liệu được lưu tại `dataset/` với ba tập `train`, `val`, `test`. Mỗi tập chứa ba thư mục `standing`, `sitting`, `moving`.

File `metadata/dataset_split_manifest.csv` lưu thông tin ảnh, nhãn, tập dữ liệu và đường dẫn tương ứng.

## 5. Exploratory Data Analysis

Notebook: `notebooks/dataset_analysis.ipynb`.

Các nội dung phân tích:
- Class Distribution.
- Train/Validation/Test Distribution.
- Image Dimensions.
- Aspect Ratio Distribution.
- Outlier Analysis.
- Sample Images Visualization.
- Final Dataset Summary.

Các biểu đồ được lưu trong `metadata/eda_figures/`.

## 6. Data Validation

Script kiểm tra: `src/final_dataset_validation.py`.

Kết quả:
- 450 ảnh hợp lệ.
- Không có file trùng SHA-256.
- Metadata khớp với dataset.
- 450 ảnh preprocessing thành công.
- Không phát hiện lỗi trong các bước kiểm tra kỹ thuật đã thực hiện.

## 7. Notes for CNN Training

- Sử dụng tập Train để huấn luyện.
- Sử dụng Validation để lựa chọn mô hình và điều chỉnh siêu tham số.
- Chỉ sử dụng Test để đánh giá cuối cùng.
- Áp dụng preprocessing nhất quán giữa huấn luyện và suy luận.
- Chỉ áp dụng data augmentation trên tập Train.
- Xác định và lưu cố định thứ tự class khi mã hóa nhãn.

**Giới hạn:** Các kiểm tra trùng lặp không chứng minh hoàn toàn tính độc lập theo người, video hoặc phiên chụp giữa các tập dữ liệu.

## 8. Environment Setup

### Requirements

- Python
- pip
- Virtual environment (venv)

### Installation

Clone or download the project and open the project
directory in VS Code.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Verify dependency consistency:

```powershell
python -m pip check
```

### Dataset Validation

Run the final dataset validation:

```powershell
python src/final_dataset_validation.py
```

Run the near-duplicate detection:

```powershell
python src/check_near_duplicates.py
```

### Notes

- Use the same preprocessing pipeline for training
  and inference.
- Do not modify the test set during model development.
- Keep the dataset split manifest unchanged.
- Review dataset validation reports before training.
