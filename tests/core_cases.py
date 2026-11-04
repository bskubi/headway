"""Clean functions for the core's tests (test_core.py): each is lowered, folded, run by the reference evaluator on
random inputs, and compared with what it computes itself. Chosen at the edges where a library's meaning is subtler
than its syntax (docs/core.md §1)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float, Integer
from scipy.special import expit, logsumexp
from scipy.stats import binom, norm, poisson


# reductions

def sums(x: Float[np.ndarray, "cells genes"]) -> Float[np.ndarray, "genes"]:
    return x.sum(0) + x.mean(axis=0) - np.max(x, 0) + x.min(-2)


def keepdims(x: Float[np.ndarray, "cells genes"]) -> Float[np.ndarray, "cells genes"]:
    return x - x.mean(1, keepdims=True)


def everything(x: Float[np.ndarray, "cells genes"]) -> float:
    return x.sum() / x.std()


def sample_sd(x: Float[np.ndarray, "cells genes"]) -> Float[np.ndarray, "genes"]:
    return x.std(0, ddof=1)                     # not the calculus's sd: an operator


def positions(x: Float[np.ndarray, "cells genes"]) -> Integer[np.ndarray, "genes"]:
    return np.argmax(x, axis=0)


def both_axes(x: Float[np.ndarray, "cells genes"]) -> float:
    return x.sum(axis=(0, 1))


# broadcasting

def logits(b: Float[np.ndarray, "bins"], d: Float[np.ndarray, "groups"]) -> Float[np.ndarray, "bins groups"]:
    return expit(b[:, None] + d[None, :])


def centered(x: Float[np.ndarray, "cells genes"], w: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells genes"]:
    return x - (w[:, None] * x).sum(0) / w.sum()


# products

def matrix(a: Float[np.ndarray, "bins groups"], v: Float[np.ndarray, "groups vectors"]) -> Float[np.ndarray, "bins vectors"]:
    return a @ v


def gram(x: Float[np.ndarray, "cells genes"]) -> Float[np.ndarray, "genes genes2"]:
    return x.T @ x


def gram_rows(x: Float[np.ndarray, "cells genes"]) -> Float[np.ndarray, "cells cells2"]:
    return x @ x.T


def vector(x: Float[np.ndarray, "cells genes"], w: Float[np.ndarray, "genes"]) -> Float[np.ndarray, "cells"]:
    return x @ w


def paired(a: Float[np.ndarray, "bins groups"], v: Float[np.ndarray, "kinds vectors"]) -> Float[np.ndarray, "bins vectors"]:
    return a @ v                                # groups and kinds paired by position


def outer(a: Float[np.ndarray, "cells"], b: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells cells2"]:
    return np.outer(a, b)


def einsum(a: Float[np.ndarray, "bins groups"], v: Float[np.ndarray, "groups vectors"]) -> Float[np.ndarray, "bins"]:
    return np.einsum("ij,jk->i", a, v) + np.einsum("ij,ij->i", a, a)


def dots(x: Float[np.ndarray, "cells genes"], w: Float[np.ndarray, "genes"]) -> Float[np.ndarray, "cells"]:
    return np.dot(x, w)


# indexing

def masked(x: Float[np.ndarray, "cells genes"], learns: Bool[np.ndarray, "cells"]) -> Float[np.ndarray, "genes"]:
    return x[learns].mean(0)


def pointer(x: Float[np.ndarray, "cells genes"], cols: Integer[np.ndarray, "picked"]) -> Float[np.ndarray, "cells"]:
    return x[:, cols].sum(1)


def pairs(x: Float[np.ndarray, "cells genes"], rows: Integer[np.ndarray, "k"],
          cols: Integer[np.ndarray, "k"]) -> Float[np.ndarray, "k"]:
    return x[rows, cols]


def apart(t: Float[np.ndarray, "a b c"], i: Integer[np.ndarray, "k"], j: Integer[np.ndarray, "k"]) -> Float[np.ndarray, "k b"]:
    return t[i, :, j]                            # pointers apart: numpy puts their axis first


def apart_int(t: Float[np.ndarray, "a b c"], j: Integer[np.ndarray, "k"]) -> Float[np.ndarray, "k b"]:
    return t[0, :, j]                            # an integer with a pointer, apart: the pointer's axis first


def basic(x: Float[np.ndarray, "cells genes"], k: int) -> Float[np.ndarray, "cells"]:
    y = x[:, :k]
    z = x[::-1]
    return y.sum(1) + z[:, 0] + x[..., 1] + x[2:, 0].sum()


def newaxis(x: Float[np.ndarray, "cells"]) -> Float[np.ndarray, "cells cells2"]:
    return x[:, None] - x[None, :]


def arange_pick(a: Float[np.ndarray, "vectors"]) -> Bool[np.ndarray, "vectors"]:
    return (a >= 0) | (np.arange(len(a)) == np.argmax(a))


def signs(v: Float[np.ndarray, "genes comps"]) -> Float[np.ndarray, "genes comps"]:
    return v * np.sign(v[np.argmax(np.abs(v), 0), np.arange(v.shape[1])])


# elementwise functions

def clipped(x: Float[np.ndarray, "cells genes"], lo: float) -> Float[np.ndarray, "cells genes"]:
    return np.clip(x, lo, None) + np.where(x > 0, x, 0) + np.maximum(x, lo) + np.log1p(np.abs(x)) + np.square(x)


def made(x: Float[np.ndarray, "cells genes"]) -> Float[np.ndarray, "genes"]:
    z = np.zeros(x.shape[1])
    return z + np.ones_like(x).sum(0) + np.full(x.shape[1], 2.0)


# control

def loop_sum(xs: list[float], w: float) -> float:
    total = 0
    for i in range(len(xs)):
        d = xs[i] - w
        total += d ** 2
    return np.sqrt(total)


def family(n: int, a: float) -> list[float]:
    out = []
    for k in range(n):
        out.append(a ** k)
    return out


def comprehension(a: list[Float[np.ndarray, "bins groups"]]) -> Float[np.ndarray, "groups groups2"]:
    return sum(x.T @ x for x in a)


def cases(x: float, t: float) -> float:
    if x > t:
        y = x - t
    else:
        y = 0.0
    return 2 * y


def early(p: float, e: float) -> float:
    if p < e:
        return 0.0
    if p > 1 - e:
        return 1.0
    return np.log(p)


def newton(m: Float[np.ndarray, "bins"], n: Float[np.ndarray, "bins"], steps: int) -> float:
    b = 0.0
    for step in range(steps):
        q = expit(b)
        b = b - (n * q - m).sum() / (n * q * (1 - q)).sum()
    return b


def while_loop(x: float) -> float:
    while x > 1:
        x = x / 2
    return x


@dataclass
class Fit:
    a: Float[np.ndarray, "bins"]
    s: float


def record(x: Float[np.ndarray, "bins groups"]) -> Fit:
    return Fit(x.sum(1), float(x.sum()))


def fields(f: Fit) -> Float[np.ndarray, "bins"]:
    return f.a * f.s


# linear algebra

def solve(a: Float[np.ndarray, "n m"], y: Float[np.ndarray, "n"]) -> Float[np.ndarray, "m"]:
    return np.linalg.solve(a, y) + np.linalg.inv(a) @ y * np.linalg.det(a)


# densities and draws

def likelihood(k: Integer[np.ndarray, "cells"], n: Integer[np.ndarray, "cells"], p: Float[np.ndarray, "cells"],
               x: Float[np.ndarray, "cells"], mu: float, sd: float) -> float:
    return binom.logpmf(k, n, p).sum() + norm.logpdf(x, mu, sd).sum() + poisson.logpmf(k, 3.0).sum()


def softmax_norm(x: Float[np.ndarray, "cells genes"]) -> Float[np.ndarray, "cells genes"]:
    return x - logsumexp(x, axis=1, keepdims=True)


def simulate(mu: Float[np.ndarray, "genes"], sd: float, rng: np.random.Generator) -> Float[np.ndarray, "genes"]:
    z = rng.normal(mu, sd)
    k = rng.binomial(10, expit(z))
    return z + k


# tables (polars)

import polars as pl  # noqa: E402


def table_filter(design: pl.DataFrame) -> Float[np.ndarray, "kept"]:
    return design.filter(pl.col("learns") & (pl.col("depth") > 2))["depth"].to_numpy()


def table_columns(design: pl.DataFrame) -> pl.DataFrame:
    return design.with_columns(log_depth=pl.col("depth").log(), double=pl.col("depth") * 2)


def table_groups(design: pl.DataFrame) -> pl.DataFrame:
    return design.group_by("plate").agg(pl.col("depth").sum().alias("total"), pl.col("depth").mean().alias("mean"))


def table_counts(design: pl.DataFrame) -> pl.DataFrame:
    return design.group_by("plate", "learns").len()


def diagonals(v: Float[np.ndarray, "genes"], m: Float[np.ndarray, "genes genes"]) -> Float[np.ndarray, "genes genes"]:
    return np.diag(v) + m * np.diag(m @ m.T).sum()


# the global random state, drawn from: a recorded call's rerun draws what the call drew
def random_start(n: int):  # noqa: ANN201
    return np.random.normal(size=n) * 2


def random_start_shaped(n: int):  # noqa: ANN201
    return np.random.normal(0, 1, (n, 2)) + np.random.rand(3).sum()


# a result holding functions not of a chunk
def with_link():  # noqa: ANN201
    return {"link": lambda mu, y: mu - y}


# dict and set operators: a merge, a difference (not elementwise)
def merged(x: float):  # noqa: ANN201
    d = {"a": x, "b": 2 * x}
    return d | {"b": x + 1, "c": 3 * x}


def merged_in_place(x: float):  # noqa: ANN201
    d = {"a": x}
    d |= {"c": 3 * x}
    return d


def set_difference(x: float):  # noqa: ANN201
    return {1, 2} - {1}
