from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://senti-hybrid.vercel.app"])

# === Load artifacts ===
cnn_model = tf.keras.models.load_model("model/cnn_model.h5")
tokenizer = joblib.load("preprocess/tokenizer.pkl")

# Check if rf_model exists and has content
import os
rf_model_path = "model/rf_model.pkl"
if os.path.exists(rf_model_path) and os.path.getsize(rf_model_path) > 0:
    rf_model = joblib.load(rf_model_path)
    rf_available = True
else:
    rf_model = None
    rf_available = False

# Check if tfidf_vectorizer exists
tfidf_path = "preprocess/tfidf_vectorizer.pkl"
if os.path.exists(tfidf_path):
    tfidf_vectorizer = joblib.load(tfidf_path)
    tfidf_available = True
else:
    tfidf_vectorizer = None
    tfidf_available = False

# === CNN Predict ===
def predict_cnn(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=50, padding='post')
    probs = cnn_model.predict(pad)[0]
    class_idx = np.argmax(probs)
    confidence = float(probs[class_idx])
    return int(class_idx), confidence

# === RF Predict ===
def predict_rf(text):
    tfidf = tfidf_vectorizer.transform([text])
    probs = rf_model.predict_proba(tfidf)[0]
    class_idx = np.argmax(probs)
    confidence = float(probs[class_idx])
    return int(class_idx), confidence

@app.route("/")
def home():
    return jsonify({
        "status": "up", 
        "message": "Hybrid Sentiment Classifier running ✅",
        "models_available": {
            "cnn": True,
            "rf": rf_available and tfidf_available
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        cnn_label, cnn_conf = predict_cnn(text)
        
        # Only use RF if available
        if rf_available and tfidf_available:
            rf_label, rf_conf = predict_rf(text)
            return jsonify({
                "cnn_prediction": cnn_label,
                "cnn_confidence": round(cnn_conf, 4),
                "rf_prediction": rf_label,
                "rf_confidence": round(rf_conf, 4),
                "models_available": {"cnn": True, "rf": True}
            })
        else:
            return jsonify({
                "cnn_prediction": cnn_label,
                "cnn_confidence": round(cnn_conf, 4),
                "rf_prediction": None,
                "rf_confidence": None,
                "models_available": {"cnn": True, "rf": False}
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
