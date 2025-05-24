

# Region Classification using Fine-Tuned EfficientNet-B3

This project addresses the task of region classification from image data using a convolutional neural network. Specifically, a pre-trained **EfficientNet-B3** model is fine-tuned to classify input images into their corresponding `Region_ID`s. EfficientNet-B3, originally trained on ImageNet, offers a balanced trade-off between accuracy and computational efficiency, making it well-suited for this task.

### Model and Methodology

* **Architecture**: The model leverages the EfficientNet-B3 convolutional neural network. The original classifier head is replaced with a custom classification layer tailored to predict 15 region classes.
* **Training Strategy**: The model is fine-tuned using a weighted cross-entropy loss function, where class weights are computed inversely proportional to class frequencies to mitigate class imbalance.
* **Optimization**: Training is performed using the Adam optimizer with a learning rate scheduler (`ReduceLROnPlateau`) and an early stopping criterion based on validation loss with a patience of 10 epochs.
* **Data Preprocessing**: All images are resized to 224×224 pixels and normalized using ImageNet mean and standard deviation values. Two sets of training data are prepared — one with standard transforms and one with aggressive data augmentations (e.g., random horizontal flips, rotations, and color jitter) to improve generalization.
* **Evaluation and Logging**: Validation accuracy and loss are computed at each epoch. Predicted region labels are restored to their original values and saved for further analysis. The best-performing model checkpoint (based on validation loss) is saved at the end of training.

This approach combines effective use of transfer learning, class imbalance handling, and data augmentation to produce a robust image-based region classification system.

