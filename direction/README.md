### Task: Angle Prediction
- **Model Type**: Convolutional Neural Network (CNN) using EfficientNet-B1.
- **Model Details**: Pre-trained on ImageNet, fine-tuned to predict sin(θ) and cos(θ) for angle prediction, addressing the periodicity of angles.
- **Pre-processing**: Images resized to 224x224, normalized with ImageNet mean and std ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]). Applied multiple augmentation levels (train, mild, strong, augment) with random horizontal flips (p=0.3-0.7), rotations (10°-45°), color jitter, affine translations, and random grayscale (strong only). Combined datasets using ConcatDataset.
- **Innovative Ideas**: 
  - Custom **SinCosLoss** with MSE and unit circle regularization (sin²(θ) + cos²(θ) ≈ 1) to ensure valid trigonometric outputs.
  - Predicted sin/cos instead of direct angles to handle 360° periodicity.
  - Multi-level augmentation to improve robustness to image variations.
  - Saved per-epoch submissions and model checkpoints (every 2nd epoch) for analysis and recovery.
  - Used Mean Absolute Angular Error (MAAE) for validation, accounting for angular periodicity by taking the minimum of absolute and wrapped differences.
- **Training**: Fine-tuned with Adam optimizer (lr=1e-4), ReduceLROnPlateau scheduler, and early stopping (patience=10) based on MAAE. Batch size of 16 balanced computation and memory.
