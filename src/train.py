from src.features import extract_mfcc
from src.model import TinyNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# --- Load audio ---
base_dir = "data"
classes = ["reading", "natural"]

audio_paths = []
labels = []

for label, cls in enumerate(classes):
    folder = os.path.join(base_dir, cls)
    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            audio_paths.append(os.path.join(folder, fname))
            labels.append(label)

labels = np.array(labels)
features = np.array([extract_mfcc(p) for p in audio_paths])

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

# --- Normalize ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train model ---
model = TinyNN(input_dim=X_train.shape[1], hidden_dim=8)
history = model.train(X_train, y_train, epochs=200, batch_size=8, lr=0.01)

# --- Evaluate ---
test_pred = model.predict(X_test)
test_acc = np.mean(test_pred == y_test)
print("Test accuracy:", test_acc)

# --- Save model & scaler ---
os.makedirs("models", exist_ok=True)
np.savez("models/model.npz", W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
np.savez("models/scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
print("Model and scaler saved to models/")
print(model.W1.shape, model.W2.shape)