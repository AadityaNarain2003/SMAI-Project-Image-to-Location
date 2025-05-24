import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from tqdm import tqdm

# --------------- Model Definition ---------------
class AnglePredictor(nn.Module):
    def __init__(self):
        super(AnglePredictor, self).__init__()
        self.efficientnet = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)  # Output: [sin, cos]
        )

    def forward(self, x):
        return self.efficientnet(x)

# --------------- Load Trained Model ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnglePredictor().to(device)
model.load_state_dict(torch.load("angle_predictor_b1_sincos_best.pth", map_location=device))
model.eval()

# --------------- Define Transform ---------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------- Load and Predict Test Images ---------------
test_folder = "images_test"
results = []

image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

for filename in tqdm(image_files, desc="Predicting"):
    filepath = os.path.join(test_folder, filename)
    image = Image.open(filepath).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor).cpu().numpy()[0]
        pred_sin, pred_cos = output[0], output[1]
        pred_angle = np.rad2deg(np.arctan2(pred_sin, pred_cos)) % 360.0

    # Extract numeric ID from filename and add 369
    match = re.search(r'(\d+)', filename)
    if match:
        file_id = int(match.group(1))
        new_id = file_id + 369
        results.append({'id': new_id, 'angle': pred_angle})

# --------------- Save Results ---------------
df = pd.DataFrame(results)
df = df.sort_values(by="id")  # Sort by ID
df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to test_predictions.csv")