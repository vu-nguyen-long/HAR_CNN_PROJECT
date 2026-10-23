
from pathlib import Path
from collections import Counter
import csv

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "metadata"
MANIFEST = METADATA / "dataset_split_manifest.csv"
OUTPUT = METADATA / "source_audit.csv"

# Các file metadata có thể chứa thông tin nguồn
metadata_files = [
    p for p in METADATA.glob("*.csv")
    if p.name not in {
        "dataset_split_manifest.csv",
        "source_audit.csv",
    }
]

source_index = {}

for csv_path in metadata_files:
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                continue

            if "file_name" not in reader.fieldnames:
                continue

            for row in reader:
                name = Path(row.get("file_name", "")).name
                if not name:
                    continue

                source = row.get("source", "").strip()

                if source:
                    source_index.setdefault(name, set()).add(source)

    except (OSError, UnicodeError, csv.Error) as e:
        print(f"Could not read {csv_path.name}: {e}")

records = []
counts = Counter()

with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        name = row["file_name"]
        sources = source_index.get(name, set())

        if len(sources) == 1:
            source = next(iter(sources))
            status = "KNOWN"
        elif len(sources) > 1:
            source = "; ".join(sorted(sources))
            status = "CONFLICT"
        else:
            source = "UNKNOWN"
            status = "UNKNOWN"

        records.append({
            "file_name": name,
            "label": row["label"],
            "split": row["split"],
            "source": source,
            "status": status,
        })

        counts[status] += 1

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name", "label", "split",
            "source", "status"
        ],
    )
    writer.writeheader()
    writer.writerows(records)

print("\n===== SOURCE METADATA AUDIT =====")
print("Total images:", len(records))
print("Known sources:", counts["KNOWN"])
print("Unknown sources:", counts["UNKNOWN"])
print("Conflicting sources:", counts["CONFLICT"])
print("Report:", OUTPUT)

if len(records) != 450:
    print("\nWARNING: Expected 450 images")
else:
    print("\nSOURCE AUDIT COMPLETED")
