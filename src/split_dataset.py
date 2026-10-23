
from pathlib import Path
from collections import Counter
import random
import shutil
import csv

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "dataset_final"
OUTPUT = ROOT / "dataset"
MANIFEST = ROOT / "metadata" / "dataset_split_manifest.csv"

CLASSES = ("standing", "sitting", "moving")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

SEED = 42
TRAIN_PER_CLASS = 120
VAL_PER_CLASS = 15
TEST_PER_CLASS = 15

# Không ghi đè dữ liệu đã chia từ trước
if OUTPUT.exists():
    raise RuntimeError(
        f"Output folder already exists: {OUTPUT}\n"
        "Please inspect it before running again."
    )

if MANIFEST.exists():
    raise RuntimeError(
        f"Manifest already exists: {MANIFEST}\n"
        "Please inspect it before running again."
    )

rng = random.Random(SEED)

split_plan = {}
all_records = []

# Kiểm tra đầy đủ trước khi sao chép
for label in CLASSES:
    folder = SOURCE / label

    if not folder.is_dir():
        raise RuntimeError(f"Missing class folder: {folder}")

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if len(images) != 150:
        raise RuntimeError(
            f"{label}: expected 150 images, found {len(images)}"
        )

    rng.shuffle(images)

    split_plan[label] = {
        "train": images[:120],
        "val": images[120:135],
        "test": images[135:150],
    }

# Tạo dataset và sao chép ảnh
for label, splits in split_plan.items():
    for split, images in splits.items():
        target_dir = OUTPUT / split / label
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            destination = target_dir / image_path.name
            shutil.copy2(image_path, destination)

            all_records.append({
                "file_name": image_path.name,
                "label": label,
                "split": split,
                "source_path": str(image_path.relative_to(ROOT)),
                "output_path": str(destination.relative_to(ROOT)),
            })

# Lưu manifest để truy xuất nguồn gốc dữ liệu
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name",
            "label",
            "split",
            "source_path",
            "output_path",
        ],
    )
    writer.writeheader()
    writer.writerows(all_records)

# Kiểm tra số lượng sau khi chia
counts = Counter(
    (record["split"], record["label"])
    for record in all_records
)

print("\n===== STRATIFIED DATASET SPLIT =====")

for split in ("train", "val", "test"):
    print(f"\n{split.upper()}")

    for label in CLASSES:
        count = counts[(split, label)]
        print(f"  {label}: {count}")

print("\nTotal images:", len(all_records))
print("Random seed:", SEED)
print("Dataset saved:", OUTPUT)
print("Manifest saved:", MANIFEST)

assert len(all_records) == 450

for label in CLASSES:
    assert counts[("train", label)] == TRAIN_PER_CLASS
    assert counts[("val", label)] == VAL_PER_CLASS
    assert counts[("test", label)] == TEST_PER_CLASS

# Đảm bảo mỗi file nguồn chỉ được chia một lần
source_paths = [r["source_path"] for r in all_records]
assert len(source_paths) == len(set(source_paths))

print("\nSTATUS: SUCCESS")
