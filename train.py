from pathlib import Path
import json
import copy

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ========================================
# 基本設定
# ========================================

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("wheel_model.pth")
CLASS_NAMES_PATH = Path("class_names.json")
GRAPH_PATH = Path("training_history.png")

IMAGE_SIZE = 128
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 20

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"使用するデバイス: {DEVICE}")


# ========================================
# 画像の前処理
# ========================================

train_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ]
)

evaluation_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ]
)


# ========================================
# データセットの読み込み
# ========================================

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=evaluation_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = train_dataset.classes
class_count = len(class_names)

print("\nクラス一覧")

for class_name, class_index in train_dataset.class_to_idx.items():
    print(f"{class_index}: {class_name}")

print(f"\n学習画像数: {len(train_dataset)}")
print(f"検証画像数: {len(val_dataset)}")
print(f"クラス数: {class_count}")

with CLASS_NAMES_PATH.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        class_names,
        file,
        ensure_ascii=False,
        indent=2
    )


# ========================================
# CNNの定義
# ========================================

class WheelCNN(nn.Module):
    def __init__(self, number_of_classes):
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

        feature_size = IMAGE_SIZE // 8

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
        result = self.classifier(features)

        return result


model = WheelCNN(class_count).to(DEVICE)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("\nCNNの構造")
print(model)


# ========================================
# 1エポック分の処理
# ========================================

def run_epoch(
    data_loader,
    training
):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_count = 0
    total_count = 0

    for images, labels in data_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)

            loss = loss_function(
                outputs,
                labels
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct_count += (
            predictions == labels
        ).sum().item()

        total_count += labels.size(0)

    average_loss = total_loss / total_count
    accuracy = correct_count / total_count

    return average_loss, accuracy


# ========================================
# 学習
# ========================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

best_val_accuracy = 0.0
best_model_state = copy.deepcopy(
    model.state_dict()
)

for epoch in range(EPOCHS):
    train_loss, train_accuracy = run_epoch(
        train_loader,
        training=True
    )

    val_loss, val_accuracy = run_epoch(
        val_loader,
        training=False
    )

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_model_state = copy.deepcopy(
            model.state_dict()
        )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| train loss: {train_loss:.4f} "
        f"| train accuracy: {train_accuracy:.3f} "
        f"| val loss: {val_loss:.4f} "
        f"| val accuracy: {val_accuracy:.3f}"
    )


# ========================================
# 最良モデルの保存
# ========================================

model.load_state_dict(best_model_state)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "best_val_accuracy": best_val_accuracy,
    },
    MODEL_PATH
)

print(
    f"\n最良検証正解率: "
    f"{best_val_accuracy:.3f}"
)

print(
    f"モデルを保存しました: "
    f"{MODEL_PATH}"
)


# ========================================
# 学習履歴のグラフ
# ========================================

epoch_numbers = range(
    1,
    EPOCHS + 1
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

axes[0].plot(
    epoch_numbers,
    history["train_loss"],
    label="Train"
)

axes[0].plot(
    epoch_numbers,
    history["val_loss"],
    label="Validation"
)

axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(
    epoch_numbers,
    history["train_accuracy"],
    label="Train"
)

axes[1].plot(
    epoch_numbers,
    history["val_accuracy"],
    label="Validation"
)

axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1)
axes[1].legend()

fig.tight_layout()
fig.savefig(
    GRAPH_PATH,
    dpi=150
)

plt.close(fig)

print(
    f"学習履歴を保存しました: "
    f"{GRAPH_PATH}"
)