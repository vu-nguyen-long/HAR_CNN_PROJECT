
from pathlib import Path
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "dataset_candidates" / "batch7_sitting"
DEST = ROOT / "dataset_raw" / "sitting"

REVIEW = ROOT / "metadata" / "batch7_sitting_review.csv"
LABELS = ROOT / "metadata" / "har_labels.csv"

DEST.mkdir(parents=True, exist_ok=True)

review_df = pd.read_csv(REVIEW)
labels_df = pd.read_csv(LABELS)

approved = review_df[
    review_df["review"] == "approved"
]

new_rows = []
copied = 0

for _, row in approved.iterrows():
    name = row["file_name"]

    src = SOURCE / name
    dst = DEST / name

    if not src.exists():
        print("Khong tim thay:", name)
        continue

    if name in labels_df["file_name"].values:
        continue

    if dst.exists():
        print("Da ton tai:", name)
        continue

    shutil.copy2(src, dst)

    new_rows.append({
        "file_name": name,
        "label": "sitting"
    })

    copied += 1

if new_rows:
    labels_df = pd.concat(
        [labels_df, pd.DataFrame(new_rows)],
        ignore_index=True
    )
    labels_df.to_csv(LABELS, index=False)

print("Da bo sung:", copied, "anh sitting")
print("Tong nhan:", len(labels_df))
print(labels_df["label"].value_counts())
