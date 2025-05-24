import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from copy import deepcopy
from tqdm.auto import trange
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import ConcatDataset

# ---------------- Dataset ---------------- #
class RegionCoordDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None, sample_percent=1.0):
        self.data = pd.read_csv(csv_file)
        if sample_percent < 1.0:
            self.data = self.data.sample(frac=sample_percent, random_state=42)
        self.image_folder = image_folder
        self.transform = transform
        self.region_id_min = self.data['Region_ID'].min()
        self.data['Region_ID'] = self.data['Region_ID'] - self.region_id_min
        # Normalize latitude and longitude
        self.lat_mean = self.data['latitude'].mean()
        self.lat_std = self.data['latitude'].std()
        self.lon_mean = self.data['longitude'].mean()
        self.lon_std = self.data['longitude'].std()
        self.data['latitude'] = (self.data['latitude'] - self.lat_mean) / self.lat_std
        self.data['longitude'] = (self.data['longitude'] - self.lon_mean) / self.lon_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_folder, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['Region_ID']
        latitude = row['latitude']
        longitude = row['longitude']
        filename_id = int(row['filename'].split('_')[1].split('.')[0])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, latitude, longitude, filename_id

    def get_region_id_min(self):
        return self.region_id_min

    def get_coord_stats(self):
        return self.lat_mean, self.lat_std, self.lon_mean, self.lon_std

# ---------------- Transforms ---------------- #
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Original augmentation
augment_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Stronger augmentation
augment_transforms_strong = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),  # Increased rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # Stronger jitter
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Mild augmentation
augment_transforms_mild = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),  # Reduced rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Milder jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- Load Dataset ---------------- #
train_sample_percent = 1.0  # Change to 0.01 for 1% or any value between 0.0 and 1.0
train_dataset = RegionCoordDataset("labels_train.csv", "images_train", transform=train_transforms, sample_percent=train_sample_percent)
augmented_train_dataset = RegionCoordDataset("labels_train.csv", "images_train", transform=augment_transforms, sample_percent=train_sample_percent)
augmented_train_dataset_strong = RegionCoordDataset("labels_train.csv", "images_train", transform=augment_transforms_strong, sample_percent=train_sample_percent)
augmented_train_dataset_mild = RegionCoordDataset("labels_train.csv", "images_train", transform=augment_transforms_mild, sample_percent=train_sample_percent)
val_dataset = RegionCoordDataset("labels_val.csv", "images_val", transform=val_transforms, sample_percent=1.0)

region_id_min = val_dataset.get_region_id_min()
lat_mean, lat_std, lon_mean, lon_std = val_dataset.get_coord_stats()

# Combine original and all augmented datasets
train_dataset = ConcatDataset([
    train_dataset,
    augmented_train_dataset,
    augmented_train_dataset_strong,
    augmented_train_dataset_mild
])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ---------------- Load Pre-trained Region Classifier ---------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

for param in region_model.parameters():
    param.requires_grad = False

# ---------------- Coordinate Regressor Model ---------------- #
class CoordRegressor(nn.Module):
    def __init__(self, region_model, num_regions):
        super(CoordRegressor, self).__init__()
        self.region_model = region_model
        self.coord_efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.coord_efficientnet.classifier[1].in_features
        self.coord_efficientnet.classifier = nn.Identity()
        self.region_embedding = nn.Embedding(num_regions, 16)
        self.fc = nn.Sequential(
            nn.Linear(in_features + 16, 512),  # coord features + region embedding
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)  # Predict latitude and longitude
        )

    def forward(self, x):
        with torch.no_grad():
            region_outputs = self.region_model(x)
            region_ids = torch.argmax(region_outputs, dim=1)
        coord_features = self.coord_efficientnet(x)
        region_embed = self.region_embedding(region_ids)
        combined = torch.cat([coord_features, region_embed], dim=1)
        return self.fc(combined)

# Initialize coordinate regressor
coord_model = CoordRegressor(region_model, num_regions=15)
coord_model = coord_model.to(device)

# ---------------- Coordinate Regressor Training ---------------- #
coord_criterion = nn.MSELoss()
optimizer = optim.Adam(coord_model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.2, verbose=True)

epochs = 50
best_val_mse = float('inf')
patience = 10
patience_counter = 0
best_coord_wts = deepcopy(coord_model.state_dict())

print(f"\nTraining Coordinate Regressor with {train_sample_percent*100}% of training data...")
for epoch in trange(epochs):
    coord_model.train()
    running_coord_loss = 0.0
    for images, labels, lats, lons, _ in train_loader:
        images = images.to(device)
        coords = torch.stack([lats, lons], dim=1).to(device, dtype=torch.float32)

        optimizer.zero_grad()
        coord_outputs = coord_model(images)
        coord_loss = coord_criterion(coord_outputs, coords)
        coord_loss.backward()
        optimizer.step()

        running_coord_loss += coord_loss.item()

    avg_coord_loss = running_coord_loss / len(train_loader)

    # Validation
    coord_model.eval()
    val_coord_loss = 0.0
    lat_mse_sum = 0.0
    lon_mse_sum = 0.0
    num_samples = 0
    submission = []
    with torch.no_grad():
        for images, _, lats, lons, filename_ids in val_loader:
            images = images.to(device)
            coords = torch.stack([lats, lons], dim=1).to(device, dtype=torch.float32)

            coord_outputs = coord_model(images)
            coord_loss = coord_criterion(coord_outputs, coords)
            val_coord_loss += coord_loss.item()

            pred_coords = coord_outputs.cpu().numpy()
            true_coords = coords.cpu().numpy()
            pred_lats = pred_coords[:, 0] * lat_std + lat_mean
            pred_lons = pred_coords[:, 1] * lon_std + lon_mean
            true_lats = true_coords[:, 0] * lat_std + lat_mean
            true_lons = true_coords[:, 1] * lon_std + lon_mean

            batch_lat_mse = ((pred_lats - true_lats) ** 2).sum()
            batch_lon_mse = ((pred_lons - true_lons) ** 2).sum()
            batch_size = len(pred_lats)
            lat_mse_sum += batch_lat_mse
            lon_mse_sum += batch_lon_mse
            num_samples += batch_size

            filename_ids = filename_ids.numpy()
            for fid, p_lat, p_lon in zip(filename_ids, pred_lats, pred_lons):
                submission.append({
                    'id': fid,
                    'latitude': p_lat,
                    'longitude': p_lon
                })

    avg_val_coord_loss = val_coord_loss / len(val_loader)
    avg_lat_mse = lat_mse_sum / num_samples
    avg_lon_mse = lon_mse_sum / num_samples
    avg_mse = 0.5 * (avg_lat_mse + avg_lon_mse)

    scheduler.step(avg_mse)  # Step the scheduler (no metric needed for StepLR)

    print(f"[Epoch {epoch+1}] Train Coord Loss: {avg_coord_loss:.4f} | Val Coord Loss: {avg_val_coord_loss:.4f} | "
          f"Val Avg MSE: {avg_mse:.4f} (Lat MSE: {avg_lat_mse:.4f}, Lon MSE: {avg_lon_mse:.4f}) | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(f"/scratch/shrikara/coord_predictions_epoch_{epoch+1}.csv", index=False)
    print(f"Saved coordinate predictions for epoch {epoch+1}.")

    if avg_mse < best_val_mse:
        best_val_mse = avg_mse
        best_coord_wts = deepcopy(coord_model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Save best coordinate model
coord_model.load_state_dict(best_coord_wts)
torch.save(coord_model.state_dict(), "  coord_regressor_best.pth")
print("Best coordinate regressor saved.")