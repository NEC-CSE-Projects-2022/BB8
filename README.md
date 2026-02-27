# 👁️ Retinal Disease Classification Using Deep Learning (OCT C8 + Grad-CAM)

A deep learning–based system for **automated multiclass classification of retinal diseases** using OCT (Optical Coherence Tomography) images from the **Retinal OCT C8 dataset**.  
The project proposes a **lightweight dual-branch CNN with residual learning** and integrates **Grad-CAM** for interpretability, aiming to build an **accurate, efficient, and clinically meaningful** medical image classification pipeline.

---

## 👥 Team Information

### Bathula Timothi  
- LinkedIn: www.linkedin.com/in/timothi-bathula-6791182a8/
- **Role & Contribution:** Project lead. Responsible for overall system design, model implementation, training pipeline development, performance evaluation, and Grad-CAM integration.

---

### Kannekanti Teja Vardhana Chari  
- LinkedIn: https://www.linkedin.com/in/teja-kannekanti/
- **Role & Contribution:** Handled dataset preparation, image preprocessing, data augmentation, and experimental setup. Assisted in model training and hyperparameter tuning.

---

### Kurra Vamsi Krishna  
- LinkedIn: https://www.linkedin.com/in/vamsikrishna-k-041989348/
- **Role & Contribution:** Conducted exploratory analysis, model comparison, result visualization, and documentation. Assisted in evaluation metrics and report preparation.

---

## 📌 Abstract

Early diagnosis of retinal diseases is essential to prevent severe vision loss; however, manual OCT analysis is time-consuming and subject to observer variability. This project presents a deep learning–based automated framework for multiclass retinal disease classification using the **Retinal OCT C8 dataset**.  

The proposed system employs a lightweight dual-branch convolutional neural network with residual connections to capture both fine-grained textures and global retinal structures. To enhance model reliability, **Grad-CAM visualization** is incorporated to highlight clinically relevant regions influencing predictions.  

Experimental results demonstrate strong and balanced performance across eight retinal disease categories, indicating the framework’s suitability for computer-aided ophthalmological screening and medical image analysis research.

---

## 🧩 About the Project

This project implements an **end-to-end OCT image classification pipeline**. The system accepts an OCT scan as input and predicts the corresponding retinal disease category.

### 🎯 Objectives

- Achieve high multiclass classification accuracy  
- Maintain lightweight and efficient architecture  
- Provide model interpretability using Grad-CAM  
- Enable reproducible medical image analysis  

---

### 🏥 Applications

- Computer-aided retinal disease screening  
- Clinical decision support for ophthalmologists  
- Medical image analysis research  
- Deployment in low-resource healthcare settings  

---

## 🔁 System Workflow

```text
Input OCT Image
→ Image Preprocessing
→ Data Augmentation
→ Dual-Branch CNN with Residual Learning
→ Multiclass Classification Output
→ Grad-CAM Heatmap Visualization
```

---

## 📊 Dataset Used

### 👉 Retinal OCT C8 Dataset

The experiments are conducted on the publicly available **Retinal OCT C8 dataset**, which contains clinically collected OCT scans representing multiple retinal conditions.

#### 🗂 Dataset Details

- **Total Images:** 21,200 OCT images  
- **Number of Classes:** 8  
- **Images per Class:** 2,650  
- **Image Type:** Grayscale OCT scans  
- **Split per Class:**
  - Train: 2,070  
  - Validation: 230  
  - Test: 350  

The dataset is balanced across all disease categories, enabling fair multiclass evaluation.

---

### 🏷️ Class Labels

- Age-related Macular Degeneration (**AMD**)  
- Choroidal Neovascularization (**CNV**)  
- Central Serous Retinopathy (**CSR**)  
- Diabetic Macular Edema (**DME**)  
- Diabetic Retinopathy (**DR**)  
- Drusen  
- Macular Hole (**MH**)  
- Normal (Healthy Retina)  

---

## 🧰 Tools & Technologies Used

- **Programming Language:** Python  
- **Framework:** PyTorch  
- **Libraries:** NumPy, OpenCV, Matplotlib, scikit-learn  

### 💻 Development Environment

- Google Colab (GPU-based training)  
- Windows 11 (local experimentation)  

---

## 🔍 Data Preprocessing & Augmentation

To ensure robust and consistent model performance, the following preprocessing steps were applied:

- Images resized to **224 × 224**  
- Pixel normalization to **[0, 1]**  
- Cropping to remove non-informative regions  
- Noise reduction during resampling  

### 🔄 Data Augmentation

Applied only to the training set:

- Horizontal flipping  
- Small-angle rotation  
- Zoom variations  
- Contrast adjustment  

The dataset was split into **training, validation, and testing** subsets to ensure proper generalization.

---

## 🧪 Model Training Information

The model was trained using supervised deep learning with a lightweight dual-branch CNN architecture.

### ⚙️ Training Configuration

- **Loss Function:** Cross Entropy Loss  
- **Optimizer:** Adam  
- **Initial Learning Rate:** 1e-4  
- **Weight Decay:** 1e-5  
- **Batch Size:** 32  
- **Epochs:** 30  
- **Learning Rate Scheduler:** Enabled  
- **Early Stopping:** Enabled  
- **Best Model Selection:** Based on highest validation F1-score  

Training was performed using a single GPU environment.

---

## 🧾 Model Evaluation

### 📈 Metrics Used

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

Evaluation was conducted on **unseen test data** to assess real-world generalization capability.

---

## 🏆 Results (Summary)

The proposed model achieved strong and balanced performance across all retinal disease categories.

### ✅ Test Performance

- **Accuracy:** 95.07%  
- **Precision:** 95.24%  
- **Recall:** 95.07%  
- **F1-score:** 95.06%  

### 🔬 Key Observations

- Minimal confusion between visually similar classes  
- Stable convergence without major overfitting  
- Lightweight architecture suitable for low-resource deployment  
- Grad-CAM highlights clinically meaningful retinal regions  

---

## 📄 Documentation

The repository includes:

- Research paper  
- Model architecture diagrams  
- Training and validation curves  
- Confusion matrix  
- Grad-CAM visualizations  
- Experimental logs  

---

## 🚀 Future Work

Potential future enhancements include:

- Multi-modal learning with fundus images  
- Mobile and edge deployment optimization  
- Self-supervised learning on unlabeled OCT data  
- Longitudinal retinal disease progression analysis  
- External dataset validation  

---

## ⚠️ Notes

- This project is intended for **academic and research purposes only**.  
- The dataset and pretrained models follow their respective licenses.  
- The system is **not a replacement for professional medical diagnosis**.

