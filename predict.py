from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import WheelCNN


MODEL_PATH = Path("wheel_model.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

CLASS_NAMES = checkpoint["class_names"]
IMAGE_SIZE = checkpoint["image_size"]


model = WheelCNN(
    number_of_classes=len(CLASS_NAMES),
    image_size=IMAGE_SIZE
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


image_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ]
)


def predict_image(
    image: Image.Image,
    top_k=3
):
    image = image.convert("RGB")
    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        top_probabilities, top_indices = (
            probabilities.topk(
                k=top_k,
                dim=1
            )
        )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):
        class_name = CLASS_NAMES[index.item()]

        results.append(
            {
                "class_name": class_name,
                "probability": probability.item()
            }
        )

    return results