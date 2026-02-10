# ======================================================
# Retinal OCT Disease Classification Web App
# FINAL VERSION (Single + Batch + History)
# ======================================================

import os
import json
import numpy as np
import cv2
import tensorflow as tf

from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ======================================================
# CONFIG
# ======================================================

UPLOAD_FOLDER = "static/uploads"
GRADCAM_FOLDER = "static/gradcam"
HISTORY_FILE = "history.json"

MAX_HISTORY = 10

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

MODEL_PATH = "retinal_c8_model.h5"
LAST_CONV_LAYER = "conv2d_7"

CONF_THRESHOLD = 0.90
ENTROPY_THRESHOLD = 1.5

CLASS_LABELS = [
    "AMD", "CNV", "CSR", "DME",
    "DR", "Drusen", "Macular Hole", "Normal"
]


# ======================================================
# INIT
# ======================================================

app = Flask(__name__)
app.secret_key = "retina_ai"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)


# ======================================================
# LOAD MODEL
# ======================================================

print("Loading TensorFlow model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")


# ======================================================
# HISTORY
# ======================================================

def load_history():
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)


def save_history(item):
    data = load_history()
    data.append(item)

    if len(data) > MAX_HISTORY:
        data = data[-MAX_HISTORY:]

    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ======================================================
# HELPERS
# ======================================================

def allowed_file(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(path):
    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    batch = np.expand_dims(arr, axis=0)
    return (arr * 255).astype("uint8"), batch


def validate_image_tf(batch):
    preds = model.predict(batch, verbose=0)[0]

    confidence = float(np.max(preds))
    entropy = -np.sum(preds * np.log(preds + 1e-8))

    return not (confidence < CONF_THRESHOLD or entropy > ENTROPY_THRESHOLD)


# ======================================================
# GRADCAM
# ======================================================

def make_gradcam(batch, idx):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_out, preds = grad_model(batch)

        # 🔥 IMPORTANT FIX (handles keras list output)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        preds = tf.convert_to_tensor(preds)

        loss = preds[:, idx]

    grads = tape.gradient(loss, conv_out)

    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]

    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)


# ======================================================
# ROUTES
# ======================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/instructions")
def instructions():
    return render_template("instructions.html")

@app.route("/history")
def history():
    return render_template("history.html", items=list(reversed(load_history())))

@app.route("/history/delete/<int:index>", methods=["POST"])
def delete_history(index):

    data = load_history()

    # because you display reversed list in UI:
    # items = list(reversed(load_history()))
    real_index = len(data) - 1 - index

    if 0 <= real_index < len(data):
        item = data.pop(real_index)

        # (Optional) delete image files from static folder
        try:
            os.remove(os.path.join("static", item["image"]))
        except Exception:
            pass

        try:
            os.remove(os.path.join("static", item["grad"]))
        except Exception:
            pass

        # save updated history
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)

        flash("✅ Deleted from history", "success")

    return redirect(url_for("history"))



# ======================================================
# SINGLE IMAGE PREDICT
# ======================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        file = request.files.get("image")

        if not file or file.filename == "":
            flash("Please upload an image!", "danger")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)

        orig, batch = preprocess(path)

        if not validate_image_tf(batch):
            flash("❌ Not a valid retinal OCT image!", "danger")
            return redirect(request.url)

        preds = model.predict(batch, verbose=0)[0]

        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        label = CLASS_LABELS[idx]

        heatmap = make_gradcam(batch, idx)
        grad_img = overlay(orig, heatmap)

        grad_name = "grad_" + filename
        grad_path = os.path.join(GRADCAM_FOLDER, grad_name)

        cv2.imwrite(grad_path, cv2.cvtColor(grad_img, cv2.COLOR_RGB2BGR))

        item = {
            "image": f"uploads/{filename}",
            "grad": f"gradcam/{grad_name}",
            "label": label,
            "confidence": round(conf * 100, 2)
        }

        save_history(item)

        return render_template("result.html", **item)

    return render_template("predict.html")


# ======================================================
# BATCH PAGE (MULTI UPLOAD)
# ======================================================

@app.route("/batch", methods=["GET", "POST"])
def batch():

    if request.method == "POST":

        files = request.files.getlist("images")

        if not files:
            flash("Please upload images!", "danger")
            return redirect(request.url)

        results = []

        for file in files:

            if not file or file.filename == "":
                continue

            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            orig, batch_img = preprocess(path)

            # ✅ VALIDATE LIKE SINGLE PREDICT
            if not validate_image_tf(batch_img):
                results.append({
                    "valid": False,
                    "image": f"uploads/{filename}",
                    "message": "❌ Not a valid retinal OCT image!"
                })
                continue

            # ✅ PREDICT ONLY IF VALID
            preds = model.predict(batch_img, verbose=0)[0]

            idx = int(np.argmax(preds))
            conf = float(np.max(preds))
            label = CLASS_LABELS[idx]

            heatmap = make_gradcam(batch_img, idx)
            grad_img = overlay(orig, heatmap)

            grad_name = "grad_" + filename
            cv2.imwrite(
                os.path.join(GRADCAM_FOLDER, grad_name),
                cv2.cvtColor(grad_img, cv2.COLOR_RGB2BGR)
            )

            item = {
                "valid": True,
                "image": f"uploads/{filename}",
                "grad": f"gradcam/{grad_name}",
                "label": label,
                "confidence": round(conf * 100, 2)
            }

            save_history(item)
            results.append(item)

        # optional message if all invalid
        if results and all(not r.get("valid", False) for r in results):
            flash("No valid retinal OCT images found in your upload.", "warning")

        return render_template("batch_result.html", results=results)

    return render_template("batch.html")

# ======================================================

if __name__ == "__main__":
    app.run(debug=True)
