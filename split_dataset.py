from pathlib import Path
import random
import shutil

SOURCE_DIR = Path("raw_dataset")
OUTPUT_DIR = Path("dataset")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

random.seed(RANDOM_SEED)

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        "raw_datasetフォルダが見つかりません。"
    )

if OUTPUT_DIR.exists():
    raise FileExistsError(
        "datasetフォルダがすでに存在します。"
        "誤って上書きしないように処理を停止しました。"
    )

for class_dir in sorted(SOURCE_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    image_files = [
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    ]

    random.shuffle(image_files)

    total = len(image_files)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    split_files = {
        "train": image_files[:train_count],
        "val": image_files[
            train_count:train_count + val_count
        ],
        "test": image_files[
            train_count + val_count:
        ],
    }

    print(f"\n{class_dir.name}: 合計{total}枚")

    for split_name, files in split_files.items():
        destination_dir = (
            OUTPUT_DIR / split_name / class_dir.name
        )
        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for source_path in files:
            destination_path = (
                destination_dir / source_path.name
            )
            shutil.copy2(
                source_path,
                destination_path
            )

        print(
            f"  {split_name}: {len(files)}枚"
        )

print("\nデータセットの分割が完了しました。")