import torch
from torch import nn


class WheelCNN(nn.Module):
    def __init__(
        self,
        number_of_classes,
        image_size=128
    ):
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