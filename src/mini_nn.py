"""A tiny 2-layer neural network written in pure NumPy.

Shows exactly what frameworks like PyTorch hide:
  forward pass -> loss -> backward pass (chain rule) -> gradient descent step.

Run:  python -m src.mini_nn
"""
import numpy as np


def make_moons_like(n=400, noise=0.15, seed=0):
    """Two interleaving half-circles (non-linearly separable) - no sklearn needed."""
    rng = np.random.default_rng(seed)
    half = n // 2
    t = rng.uniform(0, np.pi, half)
    a = np.stack([np.cos(t), np.sin(t)], axis=1)
    b = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], axis=1)
    X = np.vstack([a, b]) + rng.normal(0, noise, (2 * half, 2))
    y = np.concatenate([np.zeros(half), np.ones(half)]).astype(int)
    idx = rng.permutation(2 * half)
    return X[idx], y[idx]


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    n = y.shape[0]
    return -np.log(probs[np.arange(n), y] + 1e-12).mean()


class MiniNN:
    """Input -> Linear -> ReLU -> Linear -> Softmax."""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        # He initialisation for ReLU layers
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out))
        self.b2 = np.zeros(n_out)

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.probs = softmax(self.z2)
        return self.probs

    def loss(self, X, y):
        return cross_entropy(self.forward(X), y)

    def backward(self, y):
        """Gradients of mean cross-entropy w.r.t. every parameter (chain rule)."""
        n = y.shape[0]
        dz2 = self.probs.copy()
        dz2[np.arange(n), y] -= 1          # dL/dz2 for softmax + CE
        dz2 /= n
        grads = {
            "W2": self.a1.T @ dz2,
            "b2": dz2.sum(axis=0),
        }
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)          # ReLU derivative
        grads["W1"] = self.X.T @ dz1
        grads["b1"] = dz1.sum(axis=0)
        return grads

    def step(self, grads, lr):
        for k, p in self.params().items():
            p -= lr * grads[k]             # in-place update

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def train(model, X, y, epochs=2000, lr=0.5, verbose=False):
    history = []
    for epoch in range(epochs):
        loss = model.loss(X, y)
        grads = model.backward(y)
        model.step(grads, lr)
        history.append(loss)
        if verbose and epoch % 200 == 0:
            acc = (model.predict(X) == y).mean()
            print(f"epoch {epoch:4d}  loss {loss:.4f}  acc {acc:.3f}")
    return history


def numerical_grad(model, X, y, name, eps=1e-5):
    """Finite-difference gradient - used to *prove* backprop is correct."""
    p = model.params()[name]
    g = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = p[i]
        p[i] = old + eps
        lp = model.loss(X, y)
        p[i] = old - eps
        lm = model.loss(X, y)
        p[i] = old
        g[i] = (lp - lm) / (2 * eps)
    return g


if __name__ == "__main__":
    X, y = make_moons_like()
    model = MiniNN(2, 16, 2)
    hist = train(model, X, y, verbose=True)
    acc = (model.predict(X) == y).mean()
    print(f"final loss {hist[-1]:.4f}  train accuracy {acc:.3f}")
