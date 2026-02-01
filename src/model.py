import numpy as np

class TinyNN:
    def __init__(self, input_dim, hidden_dim=8, output_dim=1, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, 0.1, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

    # --- Activation functions ---
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # --- Forward pass ---
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        y_hat = self.sigmoid(self.z2)
        return y_hat

    # --- Loss function ---
    @staticmethod
    def binary_cross_entropy(y_hat, y):
        y = y.reshape(-1, 1)
        eps = 1e-9
        return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    # --- Backward pass ---
    def backward(self, X, y, y_hat):
        y = y.reshape(-1, 1)
        dz2 = y_hat - y
        dW2 = self.a1.T @ dz2 / X.shape[0]
        db2 = np.mean(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1 / X.shape[0]
        db1 = np.mean(dz1, axis=0)
        return dW1, db1, dW2, db2

    # --- Update weights ---
    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    # --- Training loop ---
    def train(self, X, y, epochs=200, batch_size=8, lr=0.01, verbose=True):
        history = {"loss": [], "acc": []}
        n_samples = X.shape[0]

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            X_shuf = X[idx]
            y_shuf = y[idx]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuf[i:i+batch_size]
                y_batch = y_shuf[i:i+batch_size]
                y_hat = self.forward(X_batch)
                loss = self.binary_cross_entropy(y_hat, y_batch)
                grads = self.backward(X_batch, y_batch, y_hat)
                self.step(grads, lr)

            # Evaluate epoch
            y_hat_train = self.forward(X)
            train_loss = self.binary_cross_entropy(y_hat_train, y)
            train_pred = (y_hat_train > 0.5).astype(int).flatten()
            train_acc = np.mean(train_pred == y)

            history["loss"].append(train_loss)
            history["acc"].append(train_acc)

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | acc={train_acc:.2f}")

        return history

    # --- Predict ---
    def predict(self, X):
        y_hat = self.forward(X)
        return (y_hat > 0.5).astype(int).flatten()
