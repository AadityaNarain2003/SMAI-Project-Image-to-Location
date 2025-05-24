import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torch.hub import load_state_dict_from_url
from copy import deepcopy
from PIL import Image
from tqdm.auto import trange

# ---------------- Dataset ---------------- #
class RegionDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.region_id_min = self.data['Region_ID'].min()
        self.data['Region_ID'] = self.data['Region_ID'] - self.region_id_min

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_folder, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['Region_ID']
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_region_id_min(self):
        return self.region_id_min

    def get_class_counts(self):
        """Return the count of samples per class."""
        return self.data['Region_ID'].value_counts().sort_index()

# ---------------- Transforms ---------------- #
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

augment_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------- Load Dataset ---------------- #
train_dataset = RegionDataset("labels_train.csv", "images_train", transform=train_transforms)
val_dataset = RegionDataset("labels_val.csv", "images_val", transform=val_transforms)
augmented_train_dataset = RegionDataset("labels_train.csv", "images_train", transform=augment_transforms)

region_id_min = val_dataset.get_region_id_min()

# Combine datasets for training
train_dataset = ConcatDataset([train_dataset, augmented_train_dataset])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# ---------------- Compute Class Weights ---------------- #
# Get class counts from the original training dataset (before augmentation)
class_counts = train_dataset.datasets[0].get_class_counts()  # Access the first dataset (non-augmented)
num_classes = 15  # From your model (0 to 14 after subtracting region_id_min)

# Ensure all classes are represented
class_counts = class_counts.reindex(range(num_classes), fill_value=0).values

# Compute inverse frequency weights
total_samples = sum(class_counts)
class_weights = np.array([total_samples / (num_classes * count) if count > 0 else 0 for count in class_counts])

# Normalize weights to avoid extreme values
class_weights = class_weights / class_weights.sum() * num_classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Convert to tensor and move to device
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# ---------------- Model ---------------- #


# Monkey-patch for weights loading
from torchvision.models._api import WeightsEnum
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)


in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 15)
)


model = model.to(device)

# ---------------- Training Setup ---------------- #
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Use weighted loss
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.2, verbose=True)

# ---------------- Training Loop ---------------- #
epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_model_wts = deepcopy(model.state_dict())

for epoch in trange(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    submission = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            predicted = predicted.cpu().numpy()
            for pred in predicted:
                submission.append(pred + region_id_min)  # Restore original label

    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * correct / total
    scheduler.step(avg_val_loss)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save validation predictions after each epoch
    submission_df = pd.DataFrame({'id': list(range(len(submission))), 'Region_ID': submission})
    submission_df.to_csv(f"val_predictions_epoch_{epoch+1}.csv", index=False)
    print(f"Saved validation predictions for epoch {epoch+1}.")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_wts = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# ---------------- Save Best Model ---------------- #
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "region_classifier_efficientnet_b0_best.pth")
print("Best model saved.")