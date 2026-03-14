# Project Documents

This folder contains all official documents related to the **Retinal OCT Disease Classification using Deep Learning** project, including the research paper, abstract, presentations, and detailed project documentation prepared for academic and conference evaluation.

---

# Summary

This project focuses on automated **retinal disease classification from Optical Coherence Tomography (OCT) images** using a deep learning–based architecture. The goal is to assist ophthalmologists and medical practitioners in **early detection of retinal disorders** by automatically analyzing OCT scans.

The system classifies retinal images into **eight clinically relevant categories** using a **dual-branch residual convolutional neural network**, while also providing **Grad-CAM visual explanations** to highlight the regions responsible for the prediction.

### Key Objectives

- Automated classification of retinal OCT images into multiple disease categories
- Development of a **dual-branch CNN architecture with residual learning**
- Explainable AI using **Grad-CAM visualization** for medical interpretability
- Performance evaluation using standard medical imaging metrics

---

# Repository Contents

- **CAMERA_READY_PAPER.pdf**  
  Final research paper describing the methodology, dataset, model architecture, experiments, and evaluation results.

- **PROJECT_ABSTRACT.pdf**  
  A concise summary outlining the research problem, proposed solution, and main findings.

- **COLLEGE_REVIEW_PRESENTATION.pptx**  
  Presentation used for internal academic project reviews.

- **CONFERENCE_PRESENTATION.ppt / CONFERENCE_PRESENTATION.pptx**  
  Slides prepared for conference presentation and research dissemination.

- **PROJECT_DOCUMENTATION.pdf**  
  Complete project documentation including system architecture, dataset description, preprocessing steps, model training pipeline, and experimental results.

---

# Dataset Reference

- **Dataset:** Retinal OCT C8 Dataset  
- **Classes:** 8 retinal disease categories  
- **Images:** ~21,200 OCT images  

### Categories

- AMD (Age-related Macular Degeneration)
- CNV (Choroidal Neovascularization)
- CSR (Central Serous Retinopathy)
- DME (Diabetic Macular Edema)
- DR (Diabetic Retinopathy)
- DRUSEN
- MH (Macular Hole)
- NORMAL

Images were resized and preprocessed before training to ensure consistency and improve model performance.

---

# System Description

## Input

Retinal **Optical Coherence Tomography (OCT)** scan images.

## Processing

- Image resizing to **224 × 224**
- Pixel normalization
- Data augmentation including:
  - Horizontal flipping
  - Rotation
  - Zoom transformations
- Feature extraction using deep convolutional neural networks

---

# Model Architecture

The proposed model is a **dual-branch residual CNN architecture** designed to capture both shallow and deep retinal features.

### Key Components

- Convolutional feature extractor
- Residual learning blocks
- Dual feature branches
- Global average pooling
- Feature fusion using concatenation
- Fully connected classifier with dropout

The architecture contains approximately **1.27 million trainable parameters**.

---

# Explainability

To improve interpretability for medical applications, the model integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM highlights the **important retinal regions** that contribute most to the prediction, allowing clinicians to visually verify the decision-making process of the model.

---

# Tools & Technologies

- **Programming Language:** Python 3.x
- **Framework:** TensorFlow / Keras

### Libraries

- NumPy
- OpenCV
- Matplotlib
- scikit-learn
- TensorFlow

### Development Environment

- Google Colab (GPU training)
- Local system (Windows)

---

# Training Configuration

- **Input Size:** 224 × 224
- **Batch Size:** 32
- **Optimizer:** Adam
- **Learning Rate:** 0.0005
- **Loss Function:** Categorical Cross-Entropy
- **Training Epochs:** ~20–30

### Callbacks

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

---

# Evaluation Metrics

Model performance is evaluated using standard classification metrics used in medical imaging.

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

The trained model achieved approximately **95% classification accuracy** on the test dataset.

---

# Reproducibility

For full reproducibility, refer to **PROJECT_DOCUMENTATION.pdf**, which includes:

- Dataset preprocessing pipeline
- Data split strategy
- Model architecture details
- Hyperparameter configuration
- Training procedure
- Evaluation methodology

---

# Notes

- This project is intended for **academic research and educational purposes**.
- The dataset and pretrained models are subject to their respective licenses.
- Performance may vary depending on hardware configuration and training environment.
