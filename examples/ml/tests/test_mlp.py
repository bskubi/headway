"""The perceptron learns two blobs apart."""
import torch

from mlp import steps


def data():
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (60,), generator=g)
    x = torch.randn(60, 3, generator=g) + 2.0 * y[:, None]
    return x, y


def test_train() -> None:
    x, y = data()
    g = torch.Generator().manual_seed(1)
    W1, b1 = (0.3 * torch.randn(8, 3, generator=g)).requires_grad_(), torch.zeros(8, requires_grad=True)
    W2, b2 = (0.3 * torch.randn(2, 8, generator=g)).requires_grad_(), torch.zeros(2, requires_grad=True)
    before = steps.loss(steps.forward(x, W1, b1, W2, b2), y)
    W1, b1, W2, b2 = steps.train(x, y, W1, b1, W2, b2, steps=60, lr=0.5)
    after = steps.loss(steps.forward(x, W1, b1, W2, b2), y)
    assert after < 0.5 * before
    p = steps.probabilities(steps.forward(x, W1, b1, W2, b2))
    assert torch.allclose(p.sum(1), torch.ones(60))


def test_module() -> None:
    x, _ = data()
    torch.manual_seed(0)
    model = steps.MLP(3, 8, 2)
    assert model(x).shape == (60, 2)
