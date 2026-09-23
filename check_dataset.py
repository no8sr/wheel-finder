from pathlib import Path
from PIL import Image

DATASET_DIR = Path("raw_dataset")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

total_count = 0
error_count = 0

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        "raw_datasetフォルダが見つかりません。"
        "check_dataset.pyと同じ階層に配置してください。"
    )

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    image_files = [
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"\n{class_dir.name}: {len(image_files)}枚")
    total_count += len(image_files)

    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                width, height = image.size

            if width < 200 or height < 200:
                print(
                    f"  小さい画像: {image_path.name} "
                    f"({width} x {height})"
                )

        except Exception as error:
            error_count += 1
            print(f"  読み込みエラー: {image_path.name}")
            print(f"    {error}")

print("\n確認結果")
print(f"総画像数: {total_count}枚")
print(f"読み込みエラー: {error_count}枚")