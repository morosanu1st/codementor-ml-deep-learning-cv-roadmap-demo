import numpy as np
from src.mini_nn import MiniNN, make_moons_like, train, numerical_grad, softmax


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_backprop_matches_numerical_gradient():
    X, y = make_moons_like(n=20, seed=1)
    model = MiniNN(2, 5, 2, seed=1)
    model.loss(X, y)
    grads = model.backward(y)
    for name in ["W1", "b1", "W2", "b2"]:
        num = numerical_grad(model, X, y, name)
        assert np.allclose(grads[name], num, atol=1e-6), name


def test_training_reduces_loss_and_learns_nonlinear_boundary():
    X, y = make_moons_like()
    model = MiniNN(2, 16, 2)
    hist = train(model, X, y, epochs=1500, lr=0.5)
    assert hist[-1] < hist[0] * 0.5
    assert (model.predict(X) == y).mean() > 0.95
