
# Model Development

This project implements a **deep learning model for retinal disease classification** using Optical Coherence Tomography (OCT) images.  
The model is designed to classify retinal scans into **eight disease categories** using a **dual-branch convolutional neural network with residual learning**.

---

# Model Architecture

The proposed architecture is a **Dual-Branch Residual CNN**, designed to capture both shallow and deep retinal features from OCT images.

### Key Components

- Input image size: **224 × 224**
- Initial convolution + batch normalization
- Residual learning blocks
- Two parallel feature extraction branches
- Global average pooling layers
- Feature fusion using concatenation
- Fully connected classification head
- Softmax output layer for **8 disease classes**

### Architecture Highlights

- Dual feature branches capture **multiple retinal patterns**
- Residual connections help **improve gradient flow**
- Lightweight architecture (~1.2M parameters)
- Designed for **efficient medical image classification**

---

# Data Preprocessing

Before training, OCT images undergo preprocessing to improve model performance.

### Preprocessing Steps

- Image resizing to **224 × 224**
- Pixel value normalization
- Dataset splitting:
  - Training set
  - Validation set
  - Test set

### Data Augmentation

To improve generalization, the following augmentations are applied:

- Horizontal flip
- Small rotations
- Zoom transformations

---

# Training Configuration

The model is trained using the following settings:

| Parameter | Value |
|----------|------|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 32 |
| Loss Function | Categorical Cross-Entropy |
| Epochs | ~20–30 |
| Input Size | 224 × 224 |

### Training Callbacks

The training process uses several callbacks to improve performance:

- **EarlyStopping** – stops training when validation loss stops improving  
- **ReduceLROnPlateau** – reduces learning rate when validation loss plateaus  
- **ModelCheckpoint** – saves the best performing model

---

# Model Training

Training is performed using the prepared OCT dataset.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    "Datasets/train",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical"
)
```

The model is then compiled and trained using the prepared training and validation datasets.

The training process optimizes the network weights using the **Adam optimizer** while minimizing **categorical cross-entropy loss** for multi-class classification.

Validation data is used during training to monitor model performance and prevent overfitting.

## Model Evaluation

After training, the model is evaluated on the **test dataset** to measure its performance on unseen data.

Evaluation metrics used include:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **Classification Report**

These metrics provide a comprehensive understanding of the model’s classification performance across all retinal disease categories.

---

## Single Image Prediction

The trained model supports **single OCT image prediction**.

Prediction workflow:

1. Load the trained model  
2. Resize the input OCT image to **224 × 224**  
3. Normalize pixel values  
4. Run inference using the trained model  
5. Output the predicted retinal disease class  

The model returns the predicted label among the following categories:

- AMD
- CNV
- CSR
- DME
- DR
- Drusen
- MH
- Normal

---

## Grad-CAM Visualization

To improve interpretability, the system uses **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM highlights the **important retinal regions** that influence the model's prediction, making it easier for clinicians or researchers to understand the decision-making process.

The output includes:

- Original OCT image
- Heatmap visualization
- Predicted disease class

---

## Model Performance

The trained model achieves strong performance on the retinal OCT dataset.

| Metric | Score |
|------|------|
| Accuracy | ~95% |
| Precision | ~95% |
| Recall | ~95% |
| F1 Score | ~95% |

These results demonstrate the effectiveness of the proposed architecture for retinal disease classification.

---

## Running the Notebook

To reproduce the model training:

1. Download the dataset from Kaggle
2. Place the dataset inside the `Datasets/` directory
3. Open the provided notebook
4. Run all cells sequentially

The notebook includes:

- Data preprocessing
- Model training
- Evaluation
- Prediction
- Grad-CAM visualization

---

## Dependencies

The project requires the following Python libraries:

- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

Install them using:

```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn
