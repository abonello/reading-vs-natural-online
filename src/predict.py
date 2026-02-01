import numpy as np
from src.model import TinyNN
from src.features import extract_mfcc

# Initialize model architecture
model = TinyNN(input_dim=13, hidden_dim=8)

# Load trained weights
data = np.load("models/model.npz")
model.W1 = data["W1"]
model.b1 = data["b1"]
model.W2 = data["W2"]
model.b2 = data["b2"]

def classify_audio(file_path):
    """Return 0 for Reading, 1 for Natural"""
    features = extract_mfcc(file_path)  # shape (13,)
    features = features.reshape(1, -1)  # make it (1,13)
    
    # Optional: standard scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # you could also save the scaler from training to use the same mean/std
    # for now, scale using the features themselves
    features_scaled = scaler.fit_transform(features)
    
    pred = model.predict(features_scaled)[0]
    return int(pred)
