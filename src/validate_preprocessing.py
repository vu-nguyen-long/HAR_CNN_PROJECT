
from pathlib import Path
from collections import Counter
import csv
import numpy as np

from preprocessing import preprocess_image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset_final"
OUTPUT = ROOT / "metadata" / "preprocessing_report.csv"

CLASSES = ("standing", "sitting", "moving")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

records = []
counts = Counter()

for label in CLASSES:
    folder = DATASET / label

    if not folder.is_dir():
        raise RuntimeError(f"Missing folder: {folder}")

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    for path in images:
        status = "PASS"
        error = ""
        shape = ""
        dtype = ""
        pixel_min = ""
        pixel_max = ""

        try:
            image = preprocess_image(path)

            shape = str(image.shape)
            dtype = str(image.dtype)
            pixel_min = float(np.min(image))
            pixel_max = float(np.max(image))

            assert image.shape == (224, 224, 3)
            assert image.dtype == np.float32
            assert np.isfinite(image).all()
            assert 0.0 <= pixel_min <= pixel_max <= 1.0

        except Exception as e:
            status = "FAIL"
            error = str(e)

        records.append({
            "file_name": path.name,
            "label": label,
            "status": status,
            "shape": shape,
            "dtype": dtype,
            "pixel_min": pixel_min,
            "pixel_max": pixel_max,
            "error": error
        })

        counts[(label, status)] += 1

# Save validation report
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name", "label", "status",
            "shape", "dtype",
            "pixel_min", "pixel_max", "error"
        ]
    )
    writer.writeheader()
    writer.writerows(records)

# Print summary
print("\n===== PREPROCESSING VALIDATION REPORT =====")

for label in CLASSES:
    passed = counts[(label, "PASS")]
    failed = counts[(label, "FAIL")]
    print(f"{label}: PASS={passed}, FAIL={failed}")

total_pass = sum(counts[(label, "PASS")] for label in CLASSES)
total_fail = sum(counts[(label, "FAIL")] for label in CLASSES)

print("\nTotal images:", len(records))
print("Passed:", total_pass)
print("Failed:", total_fail)
print("Report saved:", OUTPUT)

if len(records) == 450 and total_pass == 450 and total_fail == 0:
    print("\nSTATUS: SUCCESS")
else:
    print("\nSTATUS: CHECK REQUIRED")

    for record in records:
        if record["status"] == "FAIL":
            print(
                record["file_name"],
                record["label"],
                record["error"]
            )
