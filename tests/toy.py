"""A toy method for headway's tests."""
import numpy as np


def center(x):
    """Columns minus their means."""
    # step 1: the means
    mu = x.mean(0)
    # step 2: subtract
    return x - mu


def gram(x):
    return x.T @ x


SCALE = 2.0

SIZES = {
    "a": 10,
    "b": 20,
}


class Box:
    def __init__(self, x):
        self.x = x

    @property
    def total(self):
        return self.x.sum()
