
from pathlib import Path
import numpy as np
import csv

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "metadata" / "resnet_features.npz"
OUTPUT = ROOT / "metadata" / "resnet_similarity_report.csv"

THRESHOLD = 0.90
TOP_K = 30

data = np.load(INPUT)

features = data["features"]
file_names = data["file_names"]
splits = data["splits"]
labels = data["labels"]

# Các vector đã được chuẩn hóa L2
# Tích vô hướng chính là cosine similarity
similarity = features @ features.T

pairs = []

for i in range(len(features)):
    for j in range(i + 1, len(features)):

        # Chỉ so sánh giữa các tập khác nhau
        if splits[i] == splits[j]:
            continue

        pairs.append({
            "file_a": str(file_names[i]),
            "split_a": str(splits[i]),
            "label_a": str(labels[i]),
            "file_b": str(file_names[j]),
            "split_b": str(splits[j]),
            "label_b": str(labels[j]),
            "similarity": float(similarity[i, j]),
        })

pairs.sort(
    key=lambda item: item["similarity"],
    reverse=True
)

suspected = [
    p for p in pairs
    if p["similarity"] >= THRESHOLD
]

# Lưu cả các cặp vượt ngưỡng và tối thiểu
# 30 cặp đứng đầu để kiểm tra thủ công
selected = pairs[:max(TOP_K, len(suspected))]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_a", "split_a", "label_a",
            "file_b", "split_b", "label_b",
            "similarity",
        ],
    )
    writer.writeheader()
    writer.writerows(selected)

print("\n===== RESNET SIMILARITY CHECK =====")
print("Total images:", len(features))
print("Cross-split pairs checked:", len(pairs))
print("Similarity threshold:", THRESHOLD)
print("Pairs above threshold:", len(suspected))
print("Highest similarity:", round(pairs[0]["similarity"], 4))
print("Pairs saved:", len(selected))
print("Report:", OUTPUT)

print("\nTOP 10 MOST SIMILAR PAIRS")

for index, pair in enumerate(pairs[:10], 1):
    print(
        f"{index}. "
        f"{pair['file_a']} ({pair['split_a']}) "
        f"<-> "
        f"{pair['file_b']} ({pair['split_b']}) "
        f"Similarity: {pair['similarity']:.4f}"
    )

print("\nSTATUS: COMPLETED")
