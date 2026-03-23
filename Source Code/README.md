# Source Code

This folder contains the complete implementation of the **Retinal OCT Disease Classification System**, integrating both the deep learning model and the web application.

---

## Overview

The project combines **model development** and **web deployment** into a single pipeline for retinal disease detection using OCT images.

### Model Development

- Implemented using **TensorFlow / Keras**
- Designed as a **dual-branch convolutional neural network with residual learning**
- Classifies retinal OCT images into **8 disease categories**
- Includes:
  - Image preprocessing and normalization
  - Data augmentation
  - Model training and evaluation
  - Grad-CAM visualization for explainability

The trained model achieves approximately **95% accuracy** on the test dataset.

---

### Web Application

- Built using **Flask**
- Provides an interactive interface for model inference

Key features:

- Single image prediction
- Batch image prediction
- Grad-CAM visualization
- Prediction history tracking

---

## Folder Structure

```text
source_code/
│
├── app.py
├── model.ipynb
├── retinal_c8_model.h5
├── retinal_autoencoder.pth
├── threshold.npy
├── history.json
│
├── static/
│   ├── css/
│   ├── uploads/
│   ├── gradcam/
│   └── dataset_images/
│
├── templates/
└── venv/
```

---

## Workflow

1. Train the model using `model.ipynb`
2. Save the trained model (`retinal_c8_model.h5`)
3. Load the model in `app.py`
4. Upload OCT images through the web interface
5. Perform prediction using the model
6. Generate Grad-CAM visualization
7. Display and store results

---

## Technologies Used

- Python
- TensorFlow / Keras
- Flask
- OpenCV
- NumPy
- HTML / CSS
