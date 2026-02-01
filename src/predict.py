import numpy as np
from src.model import TinyNN
from src.features import extract_mfcc
from sklearn.preprocessing import StandardScaler

# Initialize model architecture
model = TinyNN(input_dim=13, hidden_dim=8)

# --- Load trained weights ---
weights = np.load("models/model.npz")
model.W1 = weights["W1"]
model.b1 = weights["b1"]
model.W2 = weights["W2"]
model.b2 = weights["b2"]

# --- Load saved scaler ---
scaler_data = np.load("models/scaler.npz")
scaler = StandardScaler()
scaler.mean_ = scaler_data["mean"]
scaler.scale_ = scaler_data["scale"]

def classify_audio(file_path):
    """Return 0 for Reading, 1 for Natural"""
    features = extract_mfcc(file_path).reshape(1, -1)  # shape (1,13)
    # features_scaled = scaler.fit_transform(features)
    features_scaled = scaler.transform(features)        # use saved scaler
    pred = model.predict(features_scaled)[0]
    return int(pred)
