"""
Spot the Fake Photo

Usage:
    python predict.py image.jpg

Output:
    Prints:
        Fraud Score
        Inference Latency (ms)

0.0 -> Real Photo
1.0 -> Photo of a Screen
"""

import sys
import time

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models, transforms


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


model = models.resnet34(weights=None)

model.fc = nn.Linear(512, 2)

model.load_state_dict(
    torch.load(
        "models/final_resnet34_detector.pth", 
        map_location=device
    )
)

model = model.to(device)

model.eval()


dummy = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    model(dummy)

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    # Original image
    img1 = transform(image)

    # Horizontal flip (Test-Time Augmentation)
    img2 = transform(
        image.transpose(Image.FLIP_LEFT_RIGHT)
    )

    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        output1 = model(img1)
        output2 = model(img2)

        prob1 = torch.softmax(output1, dim=1)[0, 1]
        prob2 = torch.softmax(output2, dim=1)[0, 1]

        score = ((prob1 + prob2) / 2).item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - start) * 1000

    return score, latency_ms


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python predict.py image.jpg")
        sys.exit(1)

    score, latency = predict(sys.argv[1])

    print(f"Fraud Score : {score:.4f}")

    print(f"Latency     : {latency:.2f} ms")

    if score >= 0.5:
        print("Prediction  : Screen Recapture")
    else:
        print("Prediction  : Real Photograph")