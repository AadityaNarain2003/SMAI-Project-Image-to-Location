import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from copy import deepcopy
from tqdm.auto import trange

# ---------------- Custom Loss for Sin/Cos ----------------
class SinCosLoss(nn.Module):
    def __init__(self, lambda_reg=0.01):
        super(SinCosLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_reg = lambda_reg  # Weight for unit circle regularization

    def forward(self, predicted, target):
        # predicted and target are shape (batch_size, 2), where [:, 0] is sin, [:, 1] is cos
        mse_loss = self.mse(predicted, target)
        
        # Regularization to enforce sin²(θ) + cos²(θ) ≈ 1
        sin_pred = predicted[:, 0]
        cos_pred = predicted[:, 1]
        unit_circle_loss = torch.mean((sin_pred**2 + cos_pred**2 - 1.0)**2)
        
        return mse_loss + self.lambda_reg * unit_circle_loss

# ---------------- Dataset ----------------
class AnglePredictionDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None, sample_percent=1.0):
        self.data = pd.read_csv(csv_file)
        if sample_percent < 1.0:
            self.data = self.data.sample(frac=sample_percent, random_state=42)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_folder, row['filename'])
        image = Image.open(img_path).convert('RGB')
        angle_deg = row['angle']  # Angle in degrees
        angle_rad = np.deg2rad(angle_deg)
        sin_cos = np.array([np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float32)
        filename_id = int(row['filename'].split('_')[1].split('.')[0])

        if self.transform:
            image = self.transform(image)

        return image, sin_cos, filename_id

# ---------------- Transforms ----------------
def get_transforms():
    return {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "mild": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "augment": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.6, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    	"strong": transforms.Compose([
        	transforms.Resize((224, 224)),
        	transforms.RandomHorizontalFlip(p=0.7),
        	transforms.RandomRotation(45),
        	transforms.ColorJitter(brightness=0.8, contrast=0.4, saturation=0.4, hue=0.2),
        	transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        	transforms.RandomGrayscale(p=0.1),
        	transforms.ToTensor(),
        	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    	])
    }
        
    

# ---------------- Load Dataset ----------------
transforms_dict = get_transforms()
train_sample_percent = 1.0

train_dataset = AnglePredictionDataset("labels_train.csv", "images_train", transform=transforms_dict["train"], sample_percent=train_sample_percent)
augmented_train_dataset = AnglePredictionDataset("labels_train.csv", "images_train", transform=transforms_dict["augment"], sample_percent=train_sample_percent)
augmented_train_dataset_strong = AnglePredictionDataset("labels_train.csv", "images_train", transform=transforms_dict["strong"], sample_percent=train_sample_percent)
augmented_train_dataset_mild = AnglePredictionDataset("labels_train.csv", "images_train", transform=transforms_dict["mild"], sample_percent=train_sample_percent)
val_dataset = AnglePredictionDataset("labels_val.csv", "images_val", transform=transforms_dict["val"], sample_percent=1.0)

train_dataset = ConcatDataset([
    train_dataset,
    augmented_train_dataset,
    augmented_train_dataset_strong,
    augmented_train_dataset_mild
])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ---------------- Model ----------------
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
            nn.Linear(128, 2)  # Output [sin(θ), cos(θ)]
        )

    def forward(self, x):
        return self.efficientnet(x)  # Shape: (batch_size, 2)

# ---------------- Training Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AnglePredictor().to(device)


criterion = SinCosLoss(lambda_reg=0.01)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.2, verbose=True)

# ---------------- Training Loop ----------------
epochs = 50
best_val_maae = float('inf')
patience = 10
patience_counter = 0
best_wts = deepcopy(model.state_dict())

print(f"\nTraining Angle Predictor with {train_sample_percent*100}% of training data...")
for epoch in trange(epochs):
    model.train()
    running_loss = 0.0
    for images, sin_cos, _ in train_loader:
        images = images.to(device)
        sin_cos = sin_cos.to(device, dtype=torch.float32)  # Shape: (batch_size, 2)

        optimizer.zero_grad()
        outputs = model(images)  # Shape: (batch_size, 2)
        loss = criterion(outputs, sin_cos)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    maae_sum = 0.0
    num_samples = 0
    submission = []

    with torch.no_grad():
        for images, sin_cos, filename_ids in val_loader:
            images = images.to(device)
            sin_cos = sin_cos.to(device, dtype=torch.float32)  # Shape: (batch_size, 2)

            outputs = model(images)  # Shape: (batch_size, 2)
            loss = criterion(outputs, sin_cos)
            val_loss += loss.item()

            # Compute MAAE by reconstructing angles
            pred_sin, pred_cos = outputs[:, 0].cpu().numpy(), outputs[:, 1].cpu().numpy()
            true_sin, true_cos = sin_cos[:, 0].cpu().numpy(), sin_cos[:, 1].cpu().numpy()
            
            # Convert to angles using atan2
            pred_angles = np.rad2deg(np.arctan2(pred_sin, pred_cos)) % 360.0
            true_angles = np.rad2deg(np.arctan2(true_sin, true_cos)) % 360.0
            
            # Compute MAAE
            batch_maae = np.minimum(np.abs(pred_angles - true_angles), 360.0 - np.abs(pred_angles - true_angles))
            batch_size = len(pred_angles)
            maae_sum += batch_maae.sum()
            num_samples += batch_size

            filename_ids = filename_ids.numpy()
            for fid, p_angle in zip(filename_ids, pred_angles):
                submission.append({'id': fid, 'angle': p_angle})

    avg_val_loss = val_loss / len(val_loader)
    avg_maae = maae_sum / num_samples
    scheduler.step(avg_maae)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Val Avg MAAE: {avg_maae:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(f"angle_predictions_epoch_{epoch+1}.csv", index=False)
    print(f"Saved angle predictions for epoch {epoch+1}.")

    # Save model checkpoint every 2nd epoch
    if (epoch + 1) % 2 == 0:
        checkpoint_path = f"angle_predictor_b1_sincos_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved model checkpoint: {checkpoint_path}")

    if avg_maae < best_val_maae:
        best_val_maae = avg_maae
        best_wts = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# ---------------- Save Best Model ----------------
model.load_state_dict(best_wts)
torch.save(model.state_dict(), "angle_predictor_b1_sincos_best.pth")
print("Best angle predictor (EfficientNet-B1 with sin/cos output) saved.")
