
from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "metadata" / "resnet_features.npz"

SPLITS = ("train", "val", "test")
CLASSES = ("standing", "sitting", "moving")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

torch.set_num_threads(4)

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

# Bỏ lớp phân loại cuối cùng
model.fc = torch.nn.Identity()
model.eval()

transform = weights.transforms()

features = []
file_names = []
splits = []
labels = []

with torch.inference_mode():
    for split in SPLITS:
        for label in CLASSES:
            folder = DATASET / split / label

            if not folder.is_dir():
                raise RuntimeError(f"Missing folder: {folder}")

            images = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in EXTENSIONS
            )

            for index, path in enumerate(images, 1):
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    image = img.convert("RGB")
                    tensor = transform(image).unsqueeze(0)

                vector = model(tensor)
                vector = vector.squeeze(0).cpu().numpy()

                # Chuẩn hóa vector để so sánh cosine
                norm = np.linalg.norm(vector)
                vector = vector / max(norm, 1e-12)

                features.append(vector)
                file_names.append(path.name)
                splits.append(split)
                labels.append(label)

                if index % 25 == 0:
                    print(
                        f"{split}/{label}: "
                        f"{index}/{len(images)}"
                    )

features = np.stack(features).astype(np.float32)

assert len(features) == 450
assert features.shape[1] == 2048

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT,
    features=features,
    file_names=np.array(file_names),
    splits=np.array(splits),
    labels=np.array(labels),
)

print("\n===== RESNET FEATURE EXTRACTION =====")
print("Total images:", len(features))
print("Feature shape:", features.shape)
print("Saved:", OUTPUT)
print("\nSTATUS: SUCCESS")
