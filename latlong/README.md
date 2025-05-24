### Task 1: Region Classification
- **Model Type**: Convolutional Neural Network (CNN) using EfficientNet-B0.
- **Model Details**: Pre-trained on ImageNet, fine-tuned for region classification with 15 output classes.
- **Pre-processing**: Images resized to 224x224, normalized with ImageNet mean and std ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]). Data augmentation included random horizontal flips, rotations (20°), and color jitter.
- **Innovative Ideas**: Custom classifier head with two linear layers (256 units, ReLU, dropout 0.4) to adapt pre-trained features for region classification. Region IDs normalized by subtracting the minimum ID for stable training.
- **Training**: Fine-tuned with Adam optimizer, cross-entropy loss, and learning rate scheduling (ReduceLROnPlateau) for optimal convergence.
- **Outcome**: Achieved robust region classification, serving as a frozen feature extractor for the coordinate regression task.

### Task 2: Coordinate Regression
- **Model Type**: CNN-based regression model combining EfficientNet-B0 backbones.
- **Model Details**: Utilized the pre-trained and fine-tuned region classifier (frozen) and a second EfficientNet-B0 (pre-trained on ImageNet, fine-tuned) for coordinate prediction. Output predicts normalized latitude and longitude.
- **Pre-processing**: Same as region classification (resize, normalization). Enhanced data augmentation with three levels (mild, original, strong) including random flips, rotations (10°-30°), color jitter, and affine translations. Combined original and augmented datasets using ConcatDataset.
- **Innovative Ideas**: Incorporated region embeddings (16-dimensional) from the frozen classifier’s predictions, concatenated with coordinate features for context-aware regression. Used MSE loss for coordinate prediction and early stopping (patience=10) to prevent overfitting. Saved predictions per epoch for analysis.
- **Training**: Trained with Adam optimizer (lr=1e-4), MSE loss, and ReduceLROnPlateau scheduler. Best model saved based on lowest validation MSE.
