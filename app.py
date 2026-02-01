from flask import Flask, request, render_template
import os
import numpy as np
from src.features import extract_mfcc
from src.model import TinyNN
from src.predict import classify_audio
from sklearn.preprocessing import StandardScaler

# --- Initialize Flask app ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load trained model weights ---
model = TinyNN(input_dim=13, hidden_dim=8)
weights = np.load("models/model.npz")
model.W1 = weights['W1']
model.b1 = weights['b1']
model.W2 = weights['W2']
model.b2 = weights['b2']

scaler = StandardScaler()
data = np.load("models/scaler.npz")
scaler.mean_ = data["mean"]
scaler.scale_ = data["scale"]

# --- Routes ---
@app.route("/")
def index():
    return """
    <h1>Reading vs Natural Audio Classifier</h1>
    <form method="POST" action="/predict" enctype="multipart/form-data">
      <input type="file" name="audio_file">
      <input type="submit" value="Upload and Predict">
    </form>
    """

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "audio_file" not in request.files:
#         return "No file uploaded", 400
#     file = request.files["audio_file"]
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Extract features and predict
#     features = extract_mfcc(filepath).reshape(1, -1)
#     pred = model.predict(features)[0]
#     label = "Natural" if pred == 1 else "Reading"
#     return f"Prediction: {label}"


@app.route("/predict", methods=["POST"])
def predict():
    if "audio_file" not in request.files:
        return "No file uploaded", 400

    file = request.files["audio_file"]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Extract features
    features = extract_mfcc(filepath).reshape(1, -1)
    features_scaled = scaler.transform(features)  # use the same scaler as training

    pred = model.predict(features_scaled)[0]
    label = "Natural" if pred == 1 else "Reading"
    return f"Prediction: {label}"


# @app.route("/predict", methods=["POST"])
# def predict():
#     f = request.files["file"]
#     path = "./temp.wav"
#     f.save(path)
#     label = classify_audio(path)
#     return "Reading" if label == 0 else "Natural"

# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
