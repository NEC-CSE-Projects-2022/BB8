# Web Application (Flask Frontend)

This project includes a **Flask-based web application** that allows users to interact with the trained retinal disease classification model through a browser interface.

The web app enables users to:

- Upload a **single OCT image** for disease prediction
- Upload **multiple images for batch prediction**
- Visualize **Grad-CAM heatmaps**
- View **prediction history**
- Browse dataset information and instructions

The application uses the trained model `retinal_c8_model.h5` to perform inference and generate Grad-CAM visualizations.

---

# Application Features

### Single Image Prediction
Users can upload a retinal OCT image and receive:

- Predicted disease class
- Confidence score
- Grad-CAM visualization highlighting important retinal regions

---

### Batch Prediction
The system supports **multi-image upload**, allowing multiple OCT scans to be analyzed simultaneously.

For each uploaded image, the system provides:

- Disease classification
- Confidence score
- Grad-CAM visualization
- Validation check for retinal OCT images

---

### Prediction History
The application stores recent predictions in **`history.json`**.

Features include:

- Viewing previous predictions
- Displaying uploaded image and Grad-CAM result
- Deleting history entries
- Limiting history size to the **latest 10 predictions**

---

### Grad-CAM Visualization
Grad-CAM highlights the **regions of the retinal OCT scan that influenced the model's prediction**.

The web interface displays:

- Original OCT image
- Heatmap overlay
- Predicted disease class

---

# Project Structure




---

# Application Routes

| Route | Description |
|------|-------------|
| `/` | Home page |
| `/dataset` | Dataset overview page |
| `/instructions` | Application usage instructions |
| `/predict` | Single image prediction |
| `/batch` | Batch image prediction |
| `/history` | View prediction history |

---

# Running the Web Application

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/retinal-oct-classification.git
cd retinal-oct-classification
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```
### 3. Activate the Virtual Environment
- Windows
  ```bash
  venv\Scripts\activate
  ```
- Linux / macOS
  ```bash
  source venv/bin/activate
  ```
### 4. Install Required Libraries
```bash
pip install flask tensorflow numpy opencv-python scikit-learn
```
### 5. Run the Flask Application
```bash
python app.py
```
### 6. Open the Application
Open your browser and go to:
```bash
http://127.0.0.1:5000
```
## File Storage Locations

Uploaded images are stored in:
```bash
static/uploads/
```

Grad-CAM generated images are stored in:

```bash
static/gradcam/
```

Dataset preview images used in the UI are stored in:

```bash
static/dataset_images/
```

---

## Prediction Validation

Before running disease prediction, uploaded images are validated using:

- **Confidence Threshold:** `0.90`
- **Entropy Threshold:** `1.5`

If an image does not meet the validation criteria, the system will reject it as **not a valid retinal OCT image**.

---

## Technologies Used

- Python
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- HTML
- CSS

---

## Future Improvements

Possible improvements for this web application include:

- Deploying the application on cloud platforms
- Adding a REST API for external applications
- Implementing user authentication
- Supporting additional retinal disease datasets
- Improving model performance using advanced architectures


