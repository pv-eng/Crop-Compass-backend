"""
Crop Compass — Real ML/DL Inference Microservice
=================================================

Serves the two models trained in the notebooks:
  - crop_model.pkl (+ label_encoder.pkl, feature_scaler.pkl)   -> /api/crop/predict
  - disease_model.h5 (+ class_indices.json)                     -> /api/disease/predict

Run locally:
    pip install -r requirements.txt
    python app.py
    -> listens on http://localhost:5001

This is a SEPARATE Python service from your existing Node/Express server (server.ts).
See ../README.md for how to call it from server.ts (with graceful fallback to the
existing rule-based logic if this service is offline — matching the pattern already
used elsewhere in your codebase).
"""

import os
import json
import base64
import io

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

from disease_info import get_disease_info

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
IMG_SIZE = (224, 224)
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Lazy-loaded model state
# ---------------------------------------------------------------------------
_state = {
    "crop_model": None,
    "label_encoder": None,
    "feature_scaler": None,
    "disease_model": None,
    "class_indices": None,
}


def load_crop_model():
    if _state["crop_model"] is None:
        _state["crop_model"] = joblib.load(os.path.join(MODELS_DIR, "crop_model.pkl"))
        _state["label_encoder"] = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
        _state["feature_scaler"] = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))
    return _state["crop_model"], _state["label_encoder"], _state["feature_scaler"]


def load_disease_model():
    if _state["disease_model"] is None:
        import tensorflow as tf  # imported lazily so /api/crop/predict works even without TF installed
        _state["disease_model"] = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, "disease_model.h5")
        )
        with open(os.path.join(MODELS_DIR, "class_indices.json")) as f:
            _state["class_indices"] = json.load(f)
    return _state["disease_model"], _state["class_indices"]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    crop_ready = os.path.exists(os.path.join(MODELS_DIR, "crop_model.pkl"))
    disease_ready = os.path.exists(os.path.join(MODELS_DIR, "disease_model.h5"))
    return jsonify({
        "status": "online",
        "cropModelReady": crop_ready,
        "diseaseModelReady": disease_ready,
    })


@app.route("/api/crop/predict", methods=["POST"])
def predict_crop():
    """
    Request JSON body (matches CropPredictionInput in src/types.ts):
      { "nitrogen": 90, "phosphorus": 42, "potassium": 43,
        "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9 }
    """
    try:
        model, le, scaler = load_crop_model()
    except FileNotFoundError:
        return jsonify({
            "error": "Crop model not found. Run notebooks/01_crop_recommendation_model.ipynb "
                     "and place crop_model.pkl, label_encoder.pkl, feature_scaler.pkl in ml_backend/models/"
        }), 503

    data = request.get_json(force=True) or {}
    required = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    row = pd.DataFrame(
        [[data["nitrogen"], data["phosphorus"], data["potassium"],
          data["temperature"], data["humidity"], data["ph"], data["rainfall"]]],
        columns=FEATURE_COLUMNS,
    )
    row_scaled = scaler.transform(row)
    probs = model.predict_proba(row_scaled)[0]

    top_idx = np.argsort(probs)[::-1][:5]
    top_crop = le.inverse_transform([top_idx[0]])[0]

    return jsonify({
        "recommendedCrop": top_crop,
        "suitabilityScore": round(float(probs[top_idx[0]]) * 100, 2),
        "alternatives": [
            {"crop": le.inverse_transform([i])[0], "score": round(float(probs[i]) * 100, 2)}
            for i in top_idx[1:]
        ],
        "isPrototype": False,
        "engineUsed": "Random Forest (trained on Kaggle Crop Recommendation Dataset)",
    })


@app.route("/api/disease/predict", methods=["POST"])
def predict_disease():
    """
    Accepts EITHER:
      - multipart/form-data with a "file" field (image upload), OR
      - JSON body: { "imageBase64": "<base64 string, with or without data: prefix>" }
    """
    try:
        from PIL import Image
        model, class_indices = load_disease_model()
    except FileNotFoundError:
        return jsonify({
            "error": "Disease model not found. Run notebooks/02_disease_detection_cnn.ipynb "
                     "and place disease_model.h5, class_indices.json in ml_backend/models/"
        }), 503

    img = None
    if "file" in request.files:
        img = Image.open(request.files["file"].stream).convert("RGB")
    else:
        data = request.get_json(silent=True) or {}
        b64 = data.get("imageBase64")
        if not b64:
            return jsonify({"error": "Provide a 'file' upload or an 'imageBase64' field"}), 400
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    img = img.resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img).astype("float32"), axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx])
    class_name = class_indices[str(top_idx)]

    info = get_disease_info(class_name)
    is_healthy = info["disease"].lower() == "healthy"

    return jsonify({
        "plantName": info["plant"],
        "diseaseStatus": "Healthy" if is_healthy else "Disease Detected",
        "diseaseName": info["disease"],
        "confidenceScore": round(confidence, 4),
        "pathogenType": info["pathogen"],
        "symptoms": info["symptoms"],
        "organicRemedies": info["organic"],
        "chemicalTreatments": info["chemical"],
        "rawClassName": class_name,
        "isPrototype": False,
        "engineUsed": "MobileNetV2 CNN (transfer learning, PlantVillage dataset)",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
