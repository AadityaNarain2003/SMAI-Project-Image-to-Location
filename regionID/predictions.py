import os
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch import nn
from PIL import Image
import pandas as pd

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNet-B3 with pretrained weights for consistent architecture
weights = EfficientNet_B3_Weights.IMAGENET1K_V1
model = efficientnet_b3(weights=weights)

# Replace the classifier to match the training script
in_features = model.classifier[1].in_features  # 1536 for EfficientNet-B3
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 15)  # 15 classes
)

# Load the saved weights
model.load_state_dict(torch.load("region_classifier_efficientnet_b0_best.pth", map_location=device))
model = model.to(device)
model.eval()

# Define image transformations (match validation transforms for B3)
transform = transforms.Compose([
    transforms.Resize((300, 300)),  # EfficientNet-B3 expects 300x300 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to the test images folder
test_folder = "images_test"

# Placeholder for region_id_min
region_id_min = 0  # Replace with actual value if known

# Prepare lists to store results
ids = []
predictions = []

# Process each image
for i in range(369):
    img_path = None
    for ext in ['jpg', 'jpeg', 'JPEG', 'png']:
        img_name = f"img_{i:04d}.{ext}"
        path = os.path.join(test_folder, img_name)
        if os.path.exists(path):
            img_path = path
            break

    if img_path is None:
        print(f"Image img_{i:04d} not found in .jpg, .jpeg, or .png format, skipping...")
        continue

    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        pred_class = predicted.item()

    # Adjust prediction with region_id_min
    adjusted_pred = pred_class + region_id_min

    # Store results
    ids.append(i + 369)
    predictions.append(adjusted_pred)

# Create a DataFrame
results = pd.DataFrame({"id": ids, "prediction": predictions})

# Save to CSV
results.to_csv("predictions.csv", index=False, header=False, sep=",")

print("Predictions saved to predictions.csv")
