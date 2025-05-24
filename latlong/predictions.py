import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn

# ---------------- Test Dataset ---------------- #
class TestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        # Get all image files with supported extensions
        self.image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.image_files.sort()  # Sort for consistent order

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        # Extract id from filename (e.g., 'img_0000.jpg' -> 0)
        filename_id = int(img_name.split('_')[1].split('.')[0])
        
        if self.transform:
            image = self.transform(image)
        
        return image, filename_id

# ---------------- Validation Dataset (for stats only) ---------------- #
class RegionCoordDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.region_id_min = self.data['Region_ID'].min()
        self.data['Region_ID'] = self.data['Region_ID'] - self.region_id_min
        # Normalize latitude and longitude
        self.lat_mean = self.data['latitude'].mean()
        self.lat_std = self.data['latitude'].std()
        self.lon_mean = self.data['longitude'].mean()
        self.lon_std = self.data['longitude'].std()

    def get_coord_stats(self):
        return self.lat_mean, self.lat_std, self.lon_mean, self.lon_std

# ---------------- Transforms ---------------- #
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- Load Test Dataset ---------------- #
test_dataset = TestDataset("images_test", transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ---------------- Load Validation Dataset for Stats ---------------- #
val_dataset = RegionCoordDataset("labels_val.csv", "images_val", transform=test_transforms)
lat_mean, lat_std, lon_mean, lon_std = val_dataset.get_coord_stats()

# ---------------- Device Setup ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Pre-trained Region Classifier ---------------- #
region_model = efficientnet_b0()
in_features = region_model.classifier[1].in_features
region_model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 15)
)
region_model.load_state_dict(torch.load("region_classifier_efficientnet_b0_best.pth"))
region_model = region_model.to(device)
region_model.eval()

# ---------------- Coordinate Regressor Model ---------------- #
class CoordRegressor(nn.Module):
    def __init__(self, region_model, num_regions):
        super(CoordRegressor, self).__init__()
        self.region_model = region_model
        self.coord_efficientnet = efficientnet_b0(weights=None)  # Weights not needed as we load state dict
        in_features = self.coord_efficientnet.classifier[1].in_features
        self.coord_efficientnet.classifier = nn.Identity()
        self.region_embedding = nn.Embedding(num_regions, 16)
        self.fc = nn.Sequential(
            nn.Linear(in_features + 16, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        with torch.no_grad():
            region_outputs = self.region_model(x)
            region_ids = torch.argmax(region_outputs, dim=1)
        coord_features = self.coord_efficientnet(x)
        region_embed = self.region_embedding(region_ids)
        combined = torch.cat([coord_features, region_embed], dim=1)
        return self.fc(combined)

# Initialize and load coordinate regressor
coord_model = CoordRegressor(region_model, num_regions=15)
coord_model.load_state_dict(torch.load("coord_regressor_best.pth"))
coord_model = coord_model.to(device)
coord_model.eval()

# ---------------- Make Predictions ---------------- #
submission = []
with torch.no_grad():
    for images, filename_ids in test_loader:
        images = images.to(device)
        filename_ids = filename_ids.numpy()

        # Coordinate regression
        coord_outputs = coord_model(images)
        pred_coords = coord_outputs.cpu().numpy()

        # Denormalize latitude and longitude
        pred_lats = pred_coords[:, 0] * lat_std + lat_mean
        pred_lons = pred_coords[:, 1] * lon_std + lon_mean

        # Store predictions
        for fid, p_lat, p_lon in zip(filename_ids, pred_lats, pred_lons):
            submission.append({
                'id': fid + 369,  # Add 1 to filename_id
                'latitude': p_lat,
                'longitude': p_lon
            })

# ---------------- Save Predictions to CSV ---------------- #
submission_df = pd.DataFrame(submission)
submission_df.to_csv("test_predictions.csv", index=False)
print("Test predictions saved to test_predictions.csv")