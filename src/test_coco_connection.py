
import requests
import certifi
from pathlib import Path

url = (
    "https://s3.amazonaws.com/"
    "images.cocodataset.org/val2017/000000039769.jpg"
)

output_path = Path("dataset_candidates/test_coco.jpg")
output_path.parent.mkdir(parents=True, exist_ok=True)

try:
    response = requests.get(
        url,
        timeout=30,
        verify=certifi.where()
    )

    response.raise_for_status()

    print("Ket noi thanh cong!")
    print("Status code:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))

    if not response.headers.get(
        "Content-Type", ""
    ).startswith("image/"):
        raise ValueError("Du lieu nhan duoc khong phai anh")

    output_path.write_bytes(response.content)

    print("Da tai anh thanh cong!")
    print("File:", output_path)

except requests.exceptions.RequestException as e:
    print("Loi ket noi:", e)

except (OSError, ValueError) as e:
    print("Loi xu ly anh:", e)
