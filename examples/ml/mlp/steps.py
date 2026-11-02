"""A small classifier in PyTorch, written as clean code: a two-layer perceptron (as functions of its weights, and
as an nn.Module), its cross-entropy loss, and gradient descent on it.

`headway trace mlp.steps -- tests/` checks that the math the viewer shows computes what this code computes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer


def forward(x: Float[torch.Tensor, "samples features"], W1: Float[torch.Tensor, "hidden features"],
            b1: Float[torch.Tensor, "hidden"], W2: Float[torch.Tensor, "classes hidden"],
            b2: Float[torch.Tensor, "classes"]) -> Float[torch.Tensor, "samples classes"]:
    """Each sample's scores for the classes: a hidden layer of rectified linear units, then a linear layer."""
    h = torch.relu(x @ W1.T + b1)
    return h @ W2.T + b2


def loss(logits: Float[torch.Tensor, "samples classes"], y: Integer[torch.Tensor, "samples"]) -> Float[torch.Tensor, ""]:
    """The mean cross-entropy of the labels under the scores' softmax."""
    return F.cross_entropy(logits, y)


def probabilities(logits: Float[torch.Tensor, "samples classes"]) -> Float[torch.Tensor, "samples classes"]:
    """Each sample's probabilities of the classes: the softmax of its scores."""
    return torch.softmax(logits, dim=1)


def train(x: Float[torch.Tensor, "samples features"], y: Integer[torch.Tensor, "samples"],
          W1: Float[torch.Tensor, "hidden features"], b1: Float[torch.Tensor, "hidden"],
          W2: Float[torch.Tensor, "classes hidden"], b2: Float[torch.Tensor, "classes"], steps: int,
          lr: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradient descent on the loss, from the given weights."""
    for step in range(steps):
        L = loss(forward(x, W1, b1, W2, b2), y)
        gW1, gb1, gW2, gb2 = torch.autograd.grad(L, (W1, b1, W2, b2))
        W1, b1, W2, b2 = W1 - lr * gW1, b1 - lr * gb1, W2 - lr * gW2, b2 - lr * gb2
    return W1, b1, W2, b2


class MLP(torch.nn.Module):
    """The same perceptron, as PyTorch's layers."""

    def __init__(self, features: int, hidden: int, classes: int):
        super().__init__()
        self.layer1 = torch.nn.Linear(features, hidden)
        self.layer2 = torch.nn.Linear(hidden, classes)

    def forward(self, x: Float[torch.Tensor, "samples features"]) -> Float[torch.Tensor, "samples classes"]:
        return self.layer2(torch.relu(self.layer1(x)))
