
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

IMAGE_SIZE = 224


def preprocess_image(image_path, normalize=True):
    """
    Read and preprocess an image.

    Output:
        Shape: (224, 224, 3)
        dtype: float32 if normalized
        Pixel range: [0, 1] if normalized
    """

    with Image.open(image_path) as img:
        # Correct EXIF orientation and convert to RGB
        img = ImageOps.exif_transpose(img).convert("RGB")

        # Resize while preserving aspect ratio
        img.thumbnail(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        # Create square canvas and center the image
        canvas = Image.new(
            "RGB",
            (IMAGE_SIZE, IMAGE_SIZE),
            (0, 0, 0)
        )

        x = (IMAGE_SIZE - img.width) // 2
        y = (IMAGE_SIZE - img.height) // 2

        canvas.paste(img, (x, y))

        # Convert to NumPy array
        array = np.asarray(canvas)

    if normalize:
        array = array.astype(np.float32) / 255.0

    return array


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    DATASET = ROOT / "dataset_final"

    extensions = {".jpg", ".jpeg", ".png"}
    classes = ("standing", "sitting", "moving")

    for label in classes:
        folder = DATASET / label
        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

        if not images:
            print(f"{label}: No images found")
            continue

        image = preprocess_image(images[0])

        print(f"\nClass: {label}")
        print(f"Image: {images[0].name}")
        print(f"Shape: {image.shape}")
        print(f"Data type: {image.dtype}")
        print(f"Pixel min: {image.min():.4f}")
        print(f"Pixel max: {image.max():.4f}")

        assert image.shape == (224, 224, 3)
        assert image.dtype == np.float32
        assert 0.0 <= image.min() <= image.max() <= 1.0

    print("\nPREPROCESSING TEST COMPLETED")
