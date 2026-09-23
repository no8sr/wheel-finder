from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASET_DIR = Path("dataset")
MODEL_PATH = Path("wheel_model.pth")
CONFUSION_MATRIX_PATH = Path("confusion_matrix.png")

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# 学習時と同じCNN
class WheelCNN(nn.Module):
    def __init__(self, number_of_classes, image_size):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        feature_size = image_size // 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                64 * feature_size * feature_size,
                128
            ),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(
                128,
                number_of_classes
            ),
        )

    def forward(self, image):
        features = self.features(image)
        return self.classifier(features)


# 保存されたモデル情報を読み込む
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]

print(f"使用デバイス: {DEVICE}")
print(f"画像サイズ: {image_size}")
print(f"クラス数: {len(class_names)}")


# テスト画像の前処理
test_transform = transforms.Compose(
    [
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ]
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=test_transform
)

if test_dataset.classes != class_names:
    raise ValueError(
        "学習時とテスト時のクラス順序が一致しません。"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# モデルを復元
model = WheelCNN(
    number_of_classes=len(class_names),
    image_size=image_size
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# 推論
correct_top1 = 0
correct_top3 = 0
total_count = 0

true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        top1_predictions = outputs.argmax(dim=1)

        top3_predictions = outputs.topk(
            k=3,
            dim=1
        ).indices

        correct_top1 += (
            top1_predictions == labels
        ).sum().item()

        correct_top3 += (
            top3_predictions == labels.unsqueeze(1)
        ).any(dim=1).sum().item()

        total_count += labels.size(0)

        true_labels.extend(
            labels.cpu().tolist()
        )

        predicted_labels.extend(
            top1_predictions.cpu().tolist()
        )


top1_accuracy = correct_top1 / total_count
top3_accuracy = correct_top3 / total_count

print("\nテスト結果")
print(f"テスト画像数: {total_count}")
print(f"Top-1正解率: {top1_accuracy:.3f}")
print(f"Top-3正解率: {top3_accuracy:.3f}")


# クラス別の評価
print("\nクラス別評価")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=3,
        zero_division=0
    )
)


# 混同行列
matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names)))
)

figure, axis = plt.subplots(
    figsize=(11, 9)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=class_names
)

display.plot(
    ax=axis,
    cmap="Blues",
    colorbar=False,
    xticks_rotation=45
)

axis.set_title("Confusion Matrix")

figure.tight_layout()

figure.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=150
)

plt.close(figure)

print(
    f"\n混同行列を保存しました: "
    f"{CONFUSION_MATRIX_PATH}"
)