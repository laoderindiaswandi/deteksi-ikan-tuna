import os
import json
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import gdown

# --- Flask setup ---
app = Flask(__name__)
os.environ["PYTHONUNBUFFERED"] = "1"

UPLOAD_FOLDER = "uploads"
DATA_FOLDER = "/data"  # Railway volume mount path
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Ensure /data exists (volume check) ---
if not os.path.exists(DATA_FOLDER):
    print("❌ Folder /data tidak tersedia. Pastikan volume di Railway sudah dikonfigurasi.")
    exit(1)

os.makedirs(DATA_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(DATA_FOLDER, "model_final.pth")
ANNOTATION_PATH = os.path.join(DATA_FOLDER, "annotations_coco_resized.json")

# --- Download helper ---
def download_if_not_exists(url, output):
    if not os.path.exists(output):
        print(f"⬇️ Downloading {output} ...")
        try:
            gdown.download(url, output, quiet=False, fuzzy=True)
        except Exception as e:
            print(f"❌ Gagal download file {output}: {e}")
            exit(1)
    else:
        print(f"✅ File {output} sudah ada")

# --- Image resize ---
def resize_image(image_path, image_sizes):
    if not image_sizes:
        return image_path
    target_width, target_height = list(image_sizes.values())[0]
    image = cv2.imread(image_path)
    if image is None:
        return None
    resized = cv2.resize(image, (target_width, target_height))
    resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + os.path.basename(image_path))
    cv2.imwrite(resized_path, resized)
    return resized_path

# --- Route: homepage ---
@app.route("/")
def home():
    return "<h1>API Deteksi Ikan Tuna (Railway)</h1>"

# --- Route: prediction ---
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    resized_path = resize_image(filepath, image_sizes)
    if resized_path is None:
        return jsonify({"error": "Failed to process image"}), 400

    image = cv2.imread(resized_path)
    outputs = predictor(image)

    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_scores = outputs["instances"].scores.cpu().numpy()

    if len(pred_classes) > 0:
        best_idx = np.argmax(pred_scores)
        results = [{
            "label": class_labels[pred_classes[best_idx]],
            "confidence": round(float(pred_scores[best_idx]) * 100, 2)
        }]
    else:
        results = []

    os.remove(filepath)
    os.remove(resized_path)
    return jsonify({"predictions": results, "message": "Deteksi selesai"})

# --- App init ---
if __name__ == "__main__":
    print("📥 Download model")
    download_if_not_exists("https://drive.google.com/uc?id=1NqsaKb6WpvzbTdrZPK3lELgkpH9pm_Pg", MODEL_PATH)

    print("📥 Download annotation")
    download_if_not_exists("https://drive.google.com/uc?id=1NVF-CMGa8FfZUYYFITUusSC8JgToOnIO", ANNOTATION_PATH)

    print("📖 Loading annotation JSON")
    try:
        with open(ANNOTATION_PATH, "r") as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"❌ Gagal membuka annotation JSON: {e}")
        exit(1)

    print("📏 Creating image_sizes map")
    image_sizes = {img["file_name"]: (img["width"], img["height"]) for img in annotations["images"]}

    print("⚙️ Configuring Detectron2")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))

    print("📦 Setting model weights")
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cpu"

    print("🏷️ Setting metadata")
    MetadataCatalog.get("tuna_dataset").thing_classes = ["ikan_tuna_segar", "ikan_tuna_tidak_segar"]
    class_labels = MetadataCatalog.get("tuna_dataset").thing_classes

    print("🎯 Creating predictor")
    try:
        predictor = DefaultPredictor(cfg)
    except Exception as e:
        print(f"❌ Gagal membuat predictor: {e}")
        exit(1)

    print("🚀 Starting Flask server...")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
