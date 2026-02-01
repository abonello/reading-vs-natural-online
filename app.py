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


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        if "audio_file" in request.files:
            file = request.files["audio_file"]
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            pred = classify_audio(filepath)
            prediction = "Natural" if pred == 1 else "Reading"
    return render_template("index.html", filename=filename, prediction=prediction)


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "audio_file" not in request.files:
#         return "No file uploaded", 400

#     file = request.files["audio_file"]
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Extract features
#     features = extract_mfcc(filepath).reshape(1, -1)
#     features_scaled = scaler.transform(features)  # use the same scaler as training

#     pred = model.predict(features_scaled)[0]
#     label = "Natural" if pred == 1 else "Reading"
#     return f"Prediction: {label}"

@app.route("/predict", methods=["POST"])
def predict():
    filename = None

    if "audio_file" not in request.files:
        return "No file uploaded", 400
    file = request.files["audio_file"]
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    # label = "Natural" if classify_audio(filepath) == 1 else "Reading"

    # pred = model.predict(features_scaled)[0]
    # label = "Natural" if pred == 1 else "Reading"
    label = "Natural" if classify_audio(filepath) == 1 else "Reading"

    # --- Memory reporting ---
    import psutil
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {mem_mb:.2f} MB")

    # return f"Prediction: {label}"
    return render_template("index.html", filename=filename, prediction=label)


# --- Run ---
if __name__ == "__main__":
    app.run(debug=True)
