
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from preprocessing import preprocess_image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset_final"

CLASSES = ["standing", "sitting", "moving"]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

fig, axes = plt.subplots(3, 2, figsize=(9, 12))

for i, label in enumerate(CLASSES):
    folder = DATASET / label

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No images in {folder}")

    path = images[0]

    with Image.open(path) as img:
        original = ImageOps.exif_transpose(img).convert("RGB")

    processed = preprocess_image(path)

    axes[i, 0].imshow(original)
    axes[i, 0].set_title(f"{label} - Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(processed)
    axes[i, 1].set_title(f"{label} - Preprocessed (224x224)")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
