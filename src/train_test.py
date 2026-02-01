from src.features import extract_mfcc
from src.model import TinyNN
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


# --- Load audio files ---
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

# --- Extract features ---
features = np.array([extract_mfcc(p) for p in audio_paths])

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

# --- Normalize ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
np.savez("models/scaler.npz", mean=scaler.mean_, scale=scaler.scale_)

# --- Initialize and train model ---
input_dim = X_train.shape[1]   # 13
hidden_dim = 8
output_dim = 1

rng = np.random.default_rng(42)
W1 = rng.normal(0, 0.1, (input_dim, hidden_dim))
b1 = np.zeros(hidden_dim)
W2 = rng.normal(0, 0.1, (hidden_dim, output_dim))
b2 = np.zeros(output_dim)

model = TinyNN(input_dim=X_train.shape[1], hidden_dim=8)
history = model.train(X_train, y_train, epochs=200, batch_size=8, lr=0.01)

# --- Evaluate on test set ---
test_pred = model.predict(X_test)
test_acc = np.mean(test_pred == y_test)
print("Test accuracy:", test_acc)



# make sure models folder exists
os.makedirs("models", exist_ok=True)

# save weights
np.savez("models/model.npz", W1=W1, b1=b1, W2=W2, b2=b2)
print("Model weights saved to models/model.npz")

# check saved weights

data = np.load("models/model.npz")
W1 = data["W1"]
b1 = data["b1"]
W2 = data["W2"]
b2 = data["b2"]

print(W1.shape, W2.shape)